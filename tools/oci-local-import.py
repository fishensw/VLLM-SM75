"""Convert a complete OCI archive to Docker format without loading it.

--full includes all layers; --parent-rootfs creates a host-dependent import.

Only an exact existing parent chain may be omitted. The complete OCI remains
the portable artifact. Does not invoke Docker or modify its internal storage.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def convert(source, output, parent, tag, *, full=False):
    if full and parent:
        raise ValueError("Full conversion cannot omit parent layers")
    if output.exists():
        raise ValueError("Refusing to overwrite import archive")
    with tarfile.open(source, "r:") as archive:
        def blob(descriptor):
            algorithm, digest = descriptor["digest"].split(":")
            if algorithm != "sha256" or len(digest) != 64:
                raise ValueError("Unsupported digest")
            member = archive.getmember("blobs/sha256/" + digest)
            if member.size != descriptor["size"] or not member.isfile():
                raise ValueError("Invalid blob size/type")
            return member

        def read(descriptor):
            data = archive.extractfile(blob(descriptor)).read()
            if "sha256:" + hashlib.sha256(data).hexdigest() != descriptor["digest"]:
                raise ValueError("Metadata digest mismatch")
            return data

        index = json.load(archive.extractfile("index.json"))
        candidates = [d for d in index["manifests"]
                      if d.get("platform", {}).get("architecture", "amd64") == "amd64"]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one amd64 image")
        manifest = json.loads(read(candidates[0]))
        config_data = read(manifest["config"])
        config = json.loads(config_data)
        layers = config["rootfs"]["diff_ids"]
        if config["architecture"] != "amd64" or config["os"] != "linux":
            raise ValueError("Expected linux/amd64")
        if (not full and not parent) or layers[:len(parent)] != parent:
            raise ValueError("Existing parent chain does not match")
        if len(layers) != len(manifest["layers"]):
            raise ValueError("Layer count mismatch")
        names = [f"layers/{i}.tar" for i in range(len(layers))]
        config_name = manifest["config"]["digest"].split(":")[1] + ".json"
        included = 0
        with tarfile.open(output, "x") as dest:
            def add(name, data):
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode = len(data), 0o644
                dest.addfile(entry, io.BytesIO(data))
            add(config_name, config_data)
            add("manifest.json", json.dumps([{"Config": config_name, "RepoTags": [tag], "Layers": names}]).encode())
            for i, descriptor in enumerate(manifest["layers"]):
                if i < len(parent):
                    continue
                member = blob(descriptor)
                entry = tarfile.TarInfo(names[i])
                entry.size, entry.mode = member.size, 0o644
                # Docker verifies uncompressed diffIDs while registering layers.
                dest.addfile(entry, archive.extractfile(member))
                included += member.size
        return {"imageId": manifest["config"]["digest"], "tag": tag,
                "existingParentLayers": len(parent), "includedLayers": len(layers) - len(parent),
                "includedBlobBytes": included, "archiveBytes": output.stat().st_size,
                "portable": full, "requiresExactExistingParentChain": parent}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--parent-rootfs", type=Path)
    mode.add_argument("--full", action="store_true", help="Include every layer for a portable Docker archive; does not load it")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    parent = [] if args.full else json.loads(args.parent_rootfs.read_text(encoding="utf-8-sig"))["Layers"]
    result = convert(args.source, args.output, parent, args.tag, full=args.full)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
