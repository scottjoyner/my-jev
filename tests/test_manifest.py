import json

from my_jev.data import dump_jsonl
from my_jev.manifest import (
    dataset_manifest,
    write_manifest,
)
from my_jev.synth import generate_synthetic_records


def test_manifest_counts_and_hashes_dataset(tmp_path):
    data = tmp_path / "data.jsonl"
    output = tmp_path / "manifest.json"
    records = generate_synthetic_records(9, seed=9)
    dump_jsonl(iter(records), data)

    manifest = write_manifest(data, output)
    loaded = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert loaded == manifest
    assert manifest["records"] == 9
    assert manifest["questions"] == 36
    assert manifest["families"] == 5
    assert len(manifest["sha256"]) == 64
    assert manifest["targets"]["hard"] > 0
    assert manifest["targets"]["soft"] > 0
    assert dataset_manifest(data)["sha256"] == manifest["sha256"]
