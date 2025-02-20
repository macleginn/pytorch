"""
This module provides the infrastructure for creating and managing sticky cache
for torch.compile. We mainly have two abstractions here:
  - Precompile: Compiled artifacts related to one particular recompile.
  - StickyCache: Overarching data structure for store and lookup a list of precompiles.

This is different from the typical global caching system in the sense that sticky cache is
always saved/loaded via user API calls. This means the caching behavior is always under
user control explicitly so that a stronger guarantee can be provided about cache hit for a
specific compiled model. Users can load the sticky cache from a different process or even
host but cautions should be taken that sticky cache will only check a subset of the original
Dynamo guards so there might be soundness problems.
"""

import contextlib
import dataclasses
import functools
import glob
import importlib
import logging
import os
import platform
import pickle
import shutil
import types
from collections.abc import Generator
from typing import Any, Callable, Optional, TYPE_CHECKING

import torch
import torch._inductor.package
from torch._inductor.codecache import extract_tensor_metadata_for_cache_key
from torch._subclasses.fake_tensor import TensorMetadata

from .bytecode_transformation import get_code_keys


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch._inductor.output_code import CompiledAOTI
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCacheEntry


# Dict type is not hashable, using values instead.
@functools.lru_cache(maxsize=1)
def _extract_inputs_metadata_cached(
    args: tuple[Any, ...], kwargs: tuple[Any, ...]
) -> list[TensorMetadata]:
    assert all(type(x) == torch.Tensor for x in args)
    assert all(type(x) == torch.Tensor for x in kwargs)
    return [extract_tensor_metadata_for_cache_key(t) for t in args + kwargs]


def _extract_inputs_metadata(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[TensorMetadata]:
    return _extract_inputs_metadata_cached(args, tuple(kwargs.values()))


class _GraphCompile:
    """
    Stores the compiled artifacts per compiled FX graph. This includes:
      - AOTI compiled kernels.
      - AOTAuograd wrappers.
    """
    def __init__(self, name: str) -> None:
        self.name = name
        self.forward_aoti: Optional[CompiledAOTI] = None
        self.backward_aoti: Optional[CompiledAOTI] = None
        self.aot_autograd = None
        self.aot_config = None

    def add_forward_aoti(self, aoti: "CompiledAOTI") -> None:
        assert self.forward_aoti is None
        self.forward_aoti = aoti

    def add_backward_aoti(self, aoti: "CompiledAOTI") -> None:
        assert self.backward_aoti is None
        self.backward_aoti = aoti

    def add_aot_autograd(self, aot_autograd: "AOTAutogradCacheEntry") -> None:
        assert self.aot_autograd is None
        self.aot_autograd = aot_autograd

    def add_aot_config(self, aot_config: "AOTConfig") -> None:
        assert self.aot_config is None
        self.aot_config = aot_config

    def update_aot_autograd_joint(self, aot_autograd: "AOTAutogradCacheEntry") -> None:
        assert self.aot_autograd is not None
        assert aot_autograd.compiled_bw is not None
        self.aot_autograd = aot_autograd

    @functools.cached_property
    def _callable(self):

        class CompiledFunction:
            def __init__(self, callback):
                self.callback = callback

            def load(self, *args, **kwargs):
                return self.callback

        entry = dataclasses.replace(
            self.aot_autograd,
            compiled_fw=CompiledFunction(self.forward_aoti)
        )
        if self.backward_aoti is not None:  # TODO Test this.
            entry = dataclasses.replace(
                entry,
                compiled_bw=CompiledFunction(self.backward_aoti)
            )

        with torch._dynamo.utils.dynamo_timed("backend_compile", log_pt2_compile_event=True):
            compiled_fn = entry.wrap_post_compile(None, self.aot_config, None)
        def forward(*args):
            return compiled_fn(list(args))
        return forward


def _load_graph_compile(load_path: str, graph_compile: _GraphCompile) -> None:
    """
    precondition: graph_compile is empty.
    """
    from torch._inductor.output_code import CompiledAOTI

    root = os.path.join(load_path, graph_compile.name)

    with open(root + ".aot_autograd", "rb") as f:
        aot_autograd = pickle.load(f)
    graph_compile.add_aot_autograd(aot_autograd)

    with open(root + ".aot_config", "rb") as f:
        from torch._functorch._aot_autograd.autograd_cache import AOTConfig
        aot_config = pickle.load(f)
        def _(*args, **kwargs):
            raise RuntimeError("NYI")
        graph_compile.add_aot_config(AOTConfig(_, _, _, {}, **aot_config))

    forward_aoti_path = root + ".forward.pt2"
    forward_aoti = torch._inductor.package.load_package(forward_aoti_path, "forward")

    def current_callable(args: tuple[Any, ...]) -> Any:
        return forward_aoti.loader.run(list(args))  # type: ignore[attr-defined]

    graph_compile.add_forward_aoti(CompiledAOTI(forward_aoti_path, current_callable))

    assert not os.path.exists(root + ".backward.pt2"), "TODO NYI"


def _save_graph_compile(save_path: str, graph_compile: _GraphCompile) -> None:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert graph_compile.aot_autograd is not None
    with open(os.path.join(save_path, f"{graph_compile.name}.aot_autograd"), "wb") as f:
        pickle.dump(graph_compile.aot_autograd, f)
    with open(os.path.join(save_path, f"{graph_compile.name}.aot_config"), "wb") as f:
        fields = [
            "num_params_buffers",
            "aot_id",
            "keep_inference_input_mutations",
            "is_export",
            "no_tangents",
            "dynamic_shapes",
            "aot_autograd_arg_pos_to_source",
            "static_input_indices",
            "enable_log",
            "pre_dispatch",
        ]
        pickle.dump({field: getattr(graph_compile.aot_config, field) for field in fields}, f)

    assert graph_compile.forward_aoti is not None
    src = graph_compile.forward_aoti.filename
    dst = os.path.join(save_path, f"{graph_compile.name}.forward.pt2")
    if isinstance(src, str) and src.endswith(".pt2"):
        # Already packed aoti file.
        shutil.copy(src, dst)
    else:
        # Unpacked aoti files.
        assert isinstance(src, list)
        torch._inductor.package.package_aoti(dst, {"forward": src})

    assert graph_compile.backward_aoti is None, "TODO NYI"


class _Precompile:
    """
    A precompile contains all the serializable information associated with a single
    compilation in torch.compile(). To restore an execution of compiled code, we will
    need to serialize the following (not exhaustive):
      - AOTI compiled code and kernels for forward and backward graph.
      - AOTAutograd wrappers for things like input mutation.
      - Dynamo bytecode for mapping Python inputs/outputs.
      - Dynamo guards for cache keys.
    """

    def __init__(self) -> None:
        self.inputs_metadata: Optional[list[TensorMetadata]] = None
        self.dynamo_code: Optional[types.CodeType] = None
        self.import_sources = {}
        self.graph_compiles = []
        self._current_graph_compile: Optional[_GraphCompile] = None

    @property
    def current_graph_compile(self) -> _GraphCompile:
        assert self._current_graph_compile is not None
        return self._current_graph_compile

    @contextlib.contextmanager
    def graph_compile_context(self, name: str) -> Generator[None, None, None]:
        """
        Set up the current graph context, should be tied to a full recompilation
        cycle.
        """
        assert self._current_graph_compile is None
        graph_compile = _GraphCompile(name)
        self._current_graph_compile = graph_compile
        try:
            yield
        finally:
            self._current_graph_compile = None
            self.graph_compiles.append(graph_compile)

    def add_import_source(self, alias: str, module_name: str) -> None:
        self.import_sources[alias] = module_name

    def add_inputs_metadata(self, inputs_metadata: list[TensorMetadata]) -> None:
        assert self.inputs_metadata is None
        self.inputs_metadata = inputs_metadata

    def add_dynamo_code(self, dynamo_code: types.CodeType) -> None:
        assert self.dynamo_code is None
        self.dynamo_code = dynamo_code

    def check_globals(self, f_globals: dict[str, Any]) -> None:
        f_globals = f_globals or {}
        global_names = set(f_globals.keys()).intersection(self.dynamo_code.co_names)
        serialized_names = {*self.import_sources.keys(), *(c.name for c in self.graph_compiles)}
        unserialized_names = global_names - serialized_names
        assert len(unserialized_names) == 0, f"Global variable names not serialized: {unserialized_names}"

    def match_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        # TODO use dynamo guards when it's ready
        assert self.inputs_metadata is not None
        for a, b in zip(self.inputs_metadata, _extract_inputs_metadata(args, kwargs)):
            if a != b:
                return False
        return True

    @functools.cached_property
    def _callable(self):
        assert self.dynamo_code is not None

        f_globals = {g.name: g._callable for g in self.graph_compiles}
        for alias, module_name in self.import_sources.items():
            f_globals[alias] = importlib.import_module(module_name)
        return types.FunctionType(self.dynamo_code, globals=f_globals)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._callable(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class SerializedCode:
    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: tuple[Any, ...]
    co_names: tuple[str, ...]
    co_varnames: tuple[str, ...]
    co_filename: str
    co_name: str
    co_firstlineno: int
    co_linetable: bytes
    co_cellvars: tuple[str, ...]
    co_freevars: tuple[str, ...]
    co_qualname: Optional[str] = None
    co_exceptiontable: Optional[bytes] = None


def _load_precompile(load_path: str, index: int) -> _Precompile:
    from torch._inductor.output_code import CompiledAOTI

    precompile = _Precompile()

    root = os.path.join(load_path, f"{index}")
    for graph_compile in glob.glob(os.path.join(root, "*.aot_autograd")):
        graph_compile_name = os.path.basename(graph_compile).rsplit(".", 1)[0]
        with precompile.graph_compile_context(graph_compile_name):
            _load_graph_compile(root, precompile.current_graph_compile)

    with open(root + ".inputs_metadata", "rb") as f:
        precompile.add_inputs_metadata(pickle.load(f))

    with open(root + ".import_sources", "rb") as f:
        import_sources = pickle.load(f)

    for alias, module_name in import_sources.items():
        precompile.add_import_source(alias, module_name)

    with open(root + ".dynamo_code", "rb") as f:
        serialized_code = pickle.load(f)
    precompile.add_dynamo_code(
        types.CodeType(*[getattr(serialized_code, key) for key in get_code_keys()])
    )

    return precompile


def _save_precompile(save_path: str, index: int, precompile: _Precompile) -> None:
    assert precompile.dynamo_code is not None
    serialized_code = SerializedCode(
        **{key: getattr(precompile.dynamo_code, key) for key in get_code_keys()}
    )
    with open(os.path.join(save_path, f"{index}.dynamo_code"), "wb") as f:
        pickle.dump(serialized_code, f)

    for graph_compile in precompile.graph_compiles:
        _save_graph_compile(os.path.join(save_path, str(index)), graph_compile)

    with open(os.path.join(save_path, f"{index}.inputs_metadata"), "wb") as f:
        pickle.dump(precompile.inputs_metadata, f)

    with open(os.path.join(save_path, f"{index}.import_sources"), "wb") as f:
        pickle.dump(precompile.import_sources, f)


class _StickyCache:
    """
    The main entry point of sticky cache system. This data structure should be created
    per torch.compile() call and propagated through the layers to collect compiled
    artifacts from Dynamo, AOTAutograd and Inductor. This essentially maintains a
    list of (guards, compiled code) which will be looked up in order when a set of
    new inputs are passed to compiled object.
    """

    def __init__(self, path: str):
        self.path = path
        if self.path.endswith(".pt2"):
            self.unimplemented("single file package")
        self._precompiles: list[_Precompile] = []
        self._current_precompile: Optional[_Precompile] = None

    @property
    def current_precompile(self) -> _Precompile:
        """
        Used to access the current precompile object within different compilation
        phases.
        """
        assert self._current_precompile is not None
        return self._current_precompile

    @contextlib.contextmanager
    def precompile_context(self) -> Generator[None, None, None]:
        """
        Set up the current precompile context, should be tied to a full recompilation
        cycle.
        """
        assert self._current_precompile is None
        precompile = _Precompile()
        self._current_precompile = precompile
        try:
            yield
        finally:
            self._current_precompile = None
            self._precompiles.append(precompile)

    def unimplemented(self, msg: str) -> None:
        raise NotImplementedError(
            f"Feature not implemented yet for sticky cache: {msg}."
        )

    def save(self) -> None:
        """
        Implementation of torch.compile().save_stikcy_cache().
        """
        assert self._current_precompile is None
        path = self.path
        if len(self._precompiles) == 0:
            logger.warning("No compiled models found for sticky cache.")
        else:
            # TODO Inductor packaging currently doesn't support things like metadata read/write,
            #      to unblock we will have a custom directory for now.
            os.makedirs(path)
            with open(os.path.join(path, "PACKAGE_INFO"), "wb") as f:
                pickle.dump({
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__
                }, f)
            for i, precompile in enumerate(self._precompiles):
                _save_precompile(path, i, precompile)

    def load(self) -> None:
        """
        Implementation of torch.compile().load_stikcy_cache().
        """
        assert self._current_precompile is None
        path = self.path
        if not os.path.exists(path) or not os.path.isdir(path):
            raise RuntimeError(f"Sticky cache path '{path}' doesn't exist.")
        # TODO Check PACKAGE_INFO
        dynamo_codes = glob.glob(os.path.join(path, "*.dynamo_code"))
        precompiles: list[_Precompile] = [_Precompile()] * len(dynamo_codes)
        for i in range(len(dynamo_codes)):
            precompiles[i] = _load_precompile(path, i)
        self._precompiles = precompiles

    def lookup(
        self, args: tuple[Any], kwargs: dict[str, Any]
    ) -> Optional[Callable[..., Any]]:
        if any(type(x) != torch.Tensor for x in args) or any(
            type(x) != torch.Tensor for x in kwargs.values()
        ):
            self.unimplemented("structured inputs")

        for precompile in self._precompiles:
            if precompile.match_inputs(args, kwargs):

                def _fn(*args: Any, **kwargs: Any) -> Any:
                    return precompile(*args, **kwargs)

                return _fn
        return None

    def reset(self, state: Optional[list[_Precompile]] = None) -> Any:
        assert self._current_precompile is None
        _precompiles = self._precompiles
        self._precompiles = state or []
        assert all(isinstance(p, _Precompile) for p in self._precompiles)
        return _precompiles
