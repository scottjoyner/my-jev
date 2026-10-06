"""The guarantees the atomic-write primitive makes, and the ones it does not.

Every run artifact in this repository -- manifests, run registry entries,
benchmark state, lease owner records, signed evidence envelopes -- is written
through `locking.atomic_write_bytes`. Before this, `atomic_write_json` had no test
whatsoever, and `harnessrouter_probe` carried two weaker private copies that had
drifted: no fsync, a pid-only temporary name that collided between two writes in one
process, and no cleanup on failure. Seven call sites used them, including the
signature envelope and the manifest bytes a verifier checks.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from my_jev.locking import atomic_write_bytes, atomic_write_json


# --- the guarantees --------------------------------------------------------


def test_it_writes_the_payload(tmp_path):
    target = tmp_path / "artifact.json"
    atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"


def test_it_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "deep" / "nested" / "artifact.json"
    atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}


def test_no_temporary_file_is_left_behind(tmp_path):
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["artifact.json"]


def test_a_failed_write_leaves_the_previous_contents_intact(tmp_path, monkeypatch):
    """The point of writing to a temporary file at all.

    Truncating the destination and then failing would destroy evidence that was
    previously good, which is worse than not writing at all.

    The failure is forced at the *rename*, not at serialisation. An unserialisable
    payload raises inside `json.dumps` before `atomic_write_bytes` is ever entered,
    so no temporary file is created and the cleanup path is never reached -- an
    earlier version of this test asserted that and passed while proving nothing
    about cleanup. Mutation testing removed the `finally` and it stayed green.
    """
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"generation": 1})

    def boom(src, dst):
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="rename failed"):
        atomic_write_json(target, {"generation": 2})

    assert json.loads(target.read_text()) == {"generation": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["artifact.json"], (
        "the temporary file was left behind"
    )


def test_an_unserialisable_payload_touches_nothing(tmp_path):
    """The earlier failure mode, asserted separately because it is a different one."""
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"generation": 1})

    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": object()})

    assert json.loads(target.read_text()) == {"generation": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["artifact.json"]


def test_the_temporary_file_is_hidden_and_will_not_match_a_json_glob(tmp_path, monkeypatch):
    """A half-written artifact must never be mistaken for a real one.

    Captured during the write, by recording the directory contents from inside
    `os.replace`.
    """
    seen: list[str] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.extend(p.name for p in Path(tmp_path).iterdir())
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    atomic_write_json(tmp_path / "manifest.json", {"ok": True})

    temporaries = [name for name in seen if name != "manifest.json"]
    assert temporaries, "expected a temporary file to exist during the write"
    assert all(name.startswith(".") for name in temporaries), temporaries
    assert not any(name.endswith(".json") for name in temporaries), temporaries


def test_the_temporary_file_lives_in_the_destination_directory(tmp_path):
    """Otherwise the final step is a cross-device move, which is not atomic."""
    target = tmp_path / "artifact.json"
    directories: list[Path] = []
    real_replace = os.replace

    def spy(src, dst):
        directories.append(Path(src).parent)
        return real_replace(src, dst)

    try:
        os.replace = spy
        atomic_write_json(target, {"ok": True})
    finally:
        os.replace = real_replace
    assert directories == [target.parent]


def test_two_writes_to_one_destination_in_the_same_process_do_not_collide(tmp_path):
    """The pid-only temporary name in the old private copies did collide here.

    Two sequential writes are fine either way, so this runs them from two threads,
    which is where a shared temporary name actually destroys one of the writes.
    """
    target = tmp_path / "artifact.json"
    payloads = [{"writer": index, "filler": "x" * 5_000} for index in range(8)]
    errors: list[BaseException] = []

    def write(payload):
        try:
            for _ in range(20):
                atomic_write_json(target, payload)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(p,)) for p in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    # Whoever won, the file is one of the complete payloads, never a blend.
    assert json.loads(target.read_text()) in payloads


def test_a_reader_never_observes_a_partial_file(tmp_path):
    """The actual guarantee, asserted while writes are in flight.

    A torn read is the failure the whole temp-file-and-rename dance exists to
    prevent, and it cannot be observed after the fact -- only by reading
    concurrently.
    """
    target = tmp_path / "artifact.json"
    small = {"payload": "s"}
    large = {"payload": "L" * 400_000}
    atomic_write_json(target, small)

    observed: list[dict] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                observed.append(json.loads(target.read_text()))
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                observed.append(exc)  # type: ignore[arg-type]

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(60):
            atomic_write_json(target, large)
            atomic_write_json(target, small)
    finally:
        stop.set()
        thread.join()

    assert observed, "the reader never ran"

    # Each read must be exactly one of the two complete payloads. Comparing against
    # the real values rather than a length is the whole point -- a torn read is
    # well-formed JSON of the wrong content, which a length check would miss.
    def describe(item):
        if isinstance(item, BaseException):
            return f"{type(item).__name__}: {item}"
        payload = item.get("payload")
        if payload == "s":
            return "complete(small)"
        if payload == "L" * 400_000:
            return "complete(large)"
        return f"TORN: keys={sorted(item)} payload_len={len(payload or '')}"

    bad = [describe(item) for item in observed if describe(item).startswith(("TORN", "JSONDecode", "FileNotFound"))]
    assert bad == [], f"observed {len(bad)} torn read(s), first: {bad[:3]}"
    assert len(observed) > 100, f"the reader only managed {len(observed)} reads"


# --- json specifics --------------------------------------------------------


def test_json_output_is_key_sorted(tmp_path):
    """These files are evidence; two equal documents should produce equal bytes."""
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"b": 1, "a": 2})
    assert target.read_text().index('"a"') < target.read_text().index('"b"')


def test_json_output_ends_with_a_newline(tmp_path):
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"a": 1})
    assert target.read_text().endswith("}\n")


def test_the_old_private_writers_are_gone(tmp_path):
    """They were a drifted duplicate of the primitive, and drift is the defect.

    Asserted rather than assumed: the two private functions in
    `harnessrouter_probe` had weaker guarantees, and the signature envelope plus
    the manifest bytes were going through them.
    """
    import my_jev.harnessrouter_probe as probe

    assert not hasattr(probe, "_atomic_json")
    assert not hasattr(probe, "_atomic_bytes")
    assert probe.atomic_write_json is atomic_write_json
    assert probe.atomic_write_bytes is atomic_write_bytes
