"""Check the installed MP runtime and parse its real CLI without starting it."""
import argparse
import hashlib
from pathlib import Path
import importlib
import importlib.metadata
import json
import sys


def probe(payload):
    lmcache_version = importlib.metadata.version("lmcache")
    vllm_version = importlib.metadata.version("vllm")
    if lmcache_version != "0.5.5" or vllm_version.split("+")[0] != "0.30.0":
        raise RuntimeError(f"需要 lmcache 0.5.5 / vllm 0.30.0，实际 {lmcache_version} / {vllm_version}")
    patch = json.loads((Path(__file__).resolve().parent.parent / "lmcache/patches/packed4d-provenance.json").read_text())
    patched_module = importlib.metadata.distribution("lmcache").locate_file(patch["module_path"])
    patch_hash = hashlib.sha256(Path(patched_module).read_bytes()).hexdigest()
    if patch_hash != patch["patched_file_sha256"]:
        raise RuntimeError("LMCache 0.5.5 必须安装本候选的 packed4D 修复；未修补版本可能恢复错误 KV，请运行随附安装器")
    import torch
    if torch.__version__.split("+")[0] != "2.13.0" or torch.version.cuda != "12.9":
        raise RuntimeError(f"需要已验证的 torch 2.13.0 / CUDA 12.9，实际 {torch.__version__} / {torch.version.cuda}")
    # LMCache otherwise silently falls back when this compiled extension is absent.
    native = importlib.import_module("lmcache.cuda_ops")
    connector = importlib.import_module("lmcache.integration.vllm.lmcache_mp_connector")
    importlib.import_module("lmcache.v1.multiprocess.http_server")
    from lmcache.cli.commands.server import ServerCommand
    from lmcache.v1.distributed.config import parse_args_to_config
    from lmcache.v1.multiprocess.config import parse_args_to_mp_server_config
    parser = argparse.ArgumentParser()
    ServerCommand().add_arguments(parser)
    parsed = parser.parse_args(payload["serverArgs"])
    storage = parse_args_to_config(parsed)
    mp = parse_args_to_mp_server_config(parsed)
    # Opaque pages must not survive an unnoticed engine/overlay upgrade. Source
    # hashes plus native-library file identities isolate normal rebuilt images.
    fingerprint = hashlib.sha256()
    for package in ("vllm", "lmcache"):
        root = Path(importlib.util.find_spec(package).origin).parent
        for file in sorted(root.rglob("*")):
            if file.is_file() and file.suffix in {".py", ".so"}:
                fingerprint.update(f"{package}/{file.relative_to(root)}".encode())
                if file.suffix == ".py":
                    fingerprint.update(file.read_bytes())
                else:
                    stat = file.stat()
                    fingerprint.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return {"ok": True, "runtimeFingerprint": fingerprint.hexdigest(), "lmcache": lmcache_version, "vllm": vllm_version,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "packed4dPatch": patch["patch_id"], "packed4dSha256": patch_hash,
            "nativeExtension": native.__file__, "connector": connector.__name__,
            "chunkSize": mp.chunk_size}


if __name__ == "__main__":
    try:
        result = probe(json.loads(sys.argv[1]))
    except (Exception, SystemExit) as error:
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    print("SM75_LMCACHE_PROBE=" + json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["ok"] else 1)
