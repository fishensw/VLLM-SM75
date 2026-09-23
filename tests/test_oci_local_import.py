import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("oci_import", Path(__file__).parents[1] / "tools/oci-local-import.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LocalImportTest(unittest.TestCase):
    def fixture(self, path):
        with tarfile.open(path, "w") as archive:
            def add(name, data):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            def blob(data):
                digest = hashlib.sha256(data).hexdigest()
                add("blobs/sha256/" + digest, data)
                return {"digest": "sha256:" + digest, "size": len(data)}
            layers = [blob(b"parent"), blob(b"new layer")]
            config = blob(json.dumps({"architecture": "amd64", "os": "linux", "rootfs": {
                "diff_ids": [d["digest"] for d in layers]}}).encode())
            manifest = blob(json.dumps({"config": config, "layers": layers}).encode())
            add("index.json", json.dumps({"manifests": [manifest]}).encode())
            return layers[0]["digest"], config["digest"]

    def test_preserves_identity_and_new_layer_only(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.tar", Path(directory) / "local.tar"
            parent, image = self.fixture(source)
            result = module.convert(source, output, [parent], "local/test:canary")
            self.assertEqual(result["imageId"], image)
            with tarfile.open(output) as archive:
                self.assertNotIn("layers/0.tar", archive.getnames())
                self.assertEqual(archive.extractfile("layers/1.tar").read(), b"new layer")
                self.assertEqual(len(json.load(archive.extractfile("manifest.json"))[0]["Layers"]), 2)
            with self.assertRaises(ValueError):
                module.convert(source, output, [parent], "local/test:canary")

    def test_full_archive_includes_every_layer_and_keeps_image_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.tar', Path(directory) / 'full.tar'
            parent, image = self.fixture(source)
            with self.assertRaises(ValueError):
                module.convert(source, output, [], 'local/test:full')
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                module.convert(source, output, [parent], 'local/test:full', full=True)
            result = module.convert(source, output, [], 'local/test:full', full=True)
            self.assertTrue(result['portable'])
            self.assertEqual(result['imageId'], image)
            self.assertEqual(result['existingParentLayers'], 0)
            with tarfile.open(output) as archive:
                manifest = json.load(archive.extractfile('manifest.json'))[0]
                self.assertEqual(archive.extractfile(manifest['Layers'][0]).read(), b'parent')
                self.assertEqual(archive.extractfile(manifest['Layers'][1]).read(), b'new layer')
                config = archive.extractfile(manifest['Config']).read()
                self.assertEqual('sha256:' + hashlib.sha256(config).hexdigest(), image)

    def test_rejects_nonmatching_parent_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.tar", Path(directory) / "local.tar"
            self.fixture(source)
            with self.assertRaises(ValueError):
                module.convert(source, output, ["sha256:" + "0" * 64], "local/test:canary")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
