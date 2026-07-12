from __future__ import annotations

import importlib

import torch.distributed as dist


def test_ordinary_python_import_does_not_patch_tcpstore() -> None:
    native_tcp_store = dist.TCPStore
    compatibility = importlib.import_module("sitecustomize")

    assert compatibility._PATCHED_FOR_TORCHRUN is False
    assert compatibility._ORIGINAL_TCP_STORE is None
    assert dist.TCPStore is native_tcp_store
