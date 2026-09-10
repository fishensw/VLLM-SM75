# SPDX-License-Identifier: Apache-2.0
"""Run explicitly in the SM75 image with a disk-backed /sleep-state mount."""


def main():
    import os

    import torch
    from vllm.device_allocator.cumem import CuMemAllocator

    from vllm.device_allocator.disk_sleep import DiskSleepBackend

    os.environ.setdefault("VLLM_AUTO_SLEEP_DISK_PATH", "/sleep-state/gpu-test")
    allocator = CuMemAllocator.get_instance()
    with allocator.use_memory_pool("weights"):
        weights = torch.arange(4 * 1024**2, dtype=torch.float32, device="cuda")
        draft = torch.arange(2 * 1024**2, dtype=torch.float32, device="cuda") * 0.5
    with allocator.use_memory_pool("kv_cache"):
        kv = torch.empty(8 * 1024**2, dtype=torch.uint8, device="cuda")
    expected = weights.cpu()
    expected_draft = draft.cpu()
    pointer = weights.data_ptr()
    output = torch.empty_like(weights)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output.copy_(weights + 1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output.copy_(weights + 1)
    backend = DiskSleepBackend()
    for cycle in range(2):
        backend.suspend(1)
        assert backend.state() == "SUSPENDED"
        snapshot = backend.snapshot
        assert snapshot.total >= weights.numel() * 4 + draft.numel() * 4
        assert (snapshot.directory / "manifest.json").is_file()
        assert all(x.is_asleep for x in allocator.pointer_to_data.values())
        backend.resume()
        assert backend.state() == "RUNNING"
        assert weights.data_ptr() == pointer
        assert torch.equal(weights.cpu(), expected)
        assert torch.equal(draft.cpu(), expected_draft)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output.cpu(), expected + 1)
        assert not snapshot.directory.exists()
        print("PASS disk GPU cycle", cycle, "runtime bytes", snapshot.total, flush=True)
    backend.suspend()
    snapshot = backend.snapshot
    path = snapshot.directory / "payload.bin"
    with open(path, "r+b") as file:
        original = file.read(1)
        file.seek(0)
        file.write(bytes([original[0] ^ 255]))
    try:
        backend.resume()
    except OSError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("Corrupt disk state was accepted")
    assert backend.state() == "SUSPENDED"
    assert snapshot.directory.exists()
    with open(path, "r+b") as file:
        file.write(original)
    backend.resume()
    assert torch.equal(weights.cpu(), expected)
    print("PASS corruption rejected, retained and successfully retried", flush=True)
    backend.root = "/dev/shm/forbidden-disk-sleep"
    try:
        backend.suspend()
    except ValueError as exc:
        assert "RAM-backed" in str(exc)
    else:
        raise AssertionError("tmpfs accepted as disk")
    assert all(not x.is_asleep for x in allocator.pointer_to_data.values())
    print("PASS rejected tmpfs without releasing model memory", flush=True)
    del graph, output, weights, draft, kv
    torch.cuda.synchronize()
    allocator.release_pools()


if __name__ == "__main__":
    main()
