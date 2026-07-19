from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import torch


ROOT = Path(
    r"C:\Users\UserY\Desktop\SCOD\RSBL-main\Dataset\COD\splits\pc_bpus"
    r"\scout_0041_0202_0404\seed2027"
)
EXPECTED = {
    41: "pc_bpus_0041_seed2027.pt",
    202: "pc_bpus_0202_seed2027.pt",
    404: "pc_bpus_0404_seed2027.pt",
}


for count, filename in EXPECTED.items():
    pt_path = ROOT / filename
    txt_path = pt_path.with_suffix(".txt")
    keys = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(keys, list) or len(keys) != count:
        raise RuntimeError(f"{pt_path}: expected a list with {count} keys")
    if not all(isinstance(key, str) and key for key in keys):
        raise RuntimeError(f"{pt_path}: keys must be non-empty strings")
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise RuntimeError(f"{pt_path}: keys must be sorted and unique")

    payload = ("\n".join(keys) + "\n").encode("utf-8")
    if txt_path.exists():
        if txt_path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite mismatched {txt_path}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{txt_path.name}.", suffix=".tmp", dir=str(ROOT)
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, txt_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    reloaded = txt_path.read_text(encoding="utf-8").splitlines()
    if reloaded != keys:
        raise RuntimeError(f"{txt_path}: TXT/PT mismatch after reload")
    print(
        f"{txt_path.name}: count={len(keys)} "
        f"sha256={hashlib.sha256(payload).hexdigest()}"
    )
