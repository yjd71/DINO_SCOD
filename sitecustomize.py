"""Narrow Windows compatibility hook for ``python -m torch.distributed.run``.

The PyTorch build used by this project does not include libuv, while its
elastic rendezvous implementation constructs ``TCPStore`` without forwarding
``use_libuv=False``.  PyTorch's normal Windows ``env://`` rendezvous already
uses the non-libuv store; this hook only fixes the parent torchrun launcher and
is inactive for ordinary Python, pytest and training worker processes.
"""

from __future__ import annotations

from functools import wraps
import sys


_ORIGINAL_TCP_STORE = None
_PATCHED_FOR_TORCHRUN = False


def _is_windows_torchrun_module() -> bool:
    original_argv = tuple(getattr(sys, "orig_argv", sys.argv))
    module_invocations = zip(original_argv, original_argv[1:])
    return sys.platform == "win32" and any(
        flag == "-m" and module == "torch.distributed.run"
        for flag, module in module_invocations
    )


if _is_windows_torchrun_module():
    try:
        import torch.distributed as _distributed
    except ModuleNotFoundError:
        # ``conda run`` itself can see the child command in sys.orig_argv while
        # executing in a base environment that does not contain torch.
        _distributed = None

    if _distributed is not None:
        _ORIGINAL_TCP_STORE = _distributed.TCPStore

        @wraps(_ORIGINAL_TCP_STORE)
        def _tcp_store_without_libuv(*args, **kwargs):
            kwargs.setdefault("use_libuv", False)
            return _ORIGINAL_TCP_STORE(*args, **kwargs)

        _distributed.TCPStore = _tcp_store_without_libuv
        _PATCHED_FOR_TORCHRUN = True
