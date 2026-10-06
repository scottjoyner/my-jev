"""Two guards against output that changes between identical runs.

Both exist because `MoveSignal.venues` was a `set[str]` and serialised differently
under every `PYTHONHASHSEED`, so two runs over identical evidence produced different
report bytes. That defeated the point of a document input, where the claim is that
anyone can re-derive a decision later from the bytes that produced it.

The instance is fixed. These are the general forms, so the next field added in the
same shape fails a test rather than a review.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "my_jev"


def _modules():
    return sorted(SRC.glob("*.py"))


def _model_fields(tree):
    """(class, field, annotation, is_private) for every pydantic model field."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any("BaseModel" in ast.unparse(b) for b in node.bases):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                yield (
                    node.name,
                    stmt.target.id,
                    ast.unparse(stmt.annotation),
                    isinstance(stmt.value, ast.Call)
                    and "PrivateAttr" in ast.unparse(stmt.value),
                )


def test_no_serialised_model_field_is_an_unordered_collection():
    """A `set` field cannot serialise reproducibly, and it did not.

    `PrivateAttr` is excluded because it is not part of the document at all -- the
    one such field in the codebase, `FleetBenchmarkAdvisory._node_ids`, never
    reaches the output.
    """
    offenders: dict[str, str] = {}
    for path in _modules():
        for cls, field, annotation, private in _model_fields(ast.parse(path.read_text())):
            if private:
                continue
            if "set[" in annotation or "Set[" in annotation or "frozenset" in annotation:
                offenders[f"{path.name}:{cls}.{field}"] = annotation

    assert offenders == {}, (
        "these fields serialise in hash order, so identical evidence would produce "
        f"different documents: {offenders}"
    )


def test_no_unordered_collection_is_iterated_without_sorting():
    """Set intersections reached the reports through `sorted()` everywhere already.

    Checked because that discipline is invisible at the call site: `for q in
    set(a) & set(b)` reads as obviously order-independent and is not, once the
    order reaches a document. Every site in the codebase wraps it, and this keeps it
    that way.
    """

    def is_sorted_call(node) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sorted"
        )

    def setish(node) -> bool:
        if isinstance(node, (ast.SetComp, ast.Set)):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("set", "frozenset")
        ):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            return setish(node.left) or setish(node.right)
        return False

    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and setish(node.iter) and not is_sorted_call(node.iter):
                offenders.append(f"{path.name}:{node.lineno} for {ast.unparse(node.iter)[:50]}")
            elif (
                isinstance(node, ast.ListComp)
                and setish(node.generators[0].iter)
                and not is_sorted_call(node.generators[0].iter)
            ):
                offenders.append(
                    f"{path.name}:{node.lineno} comprehension over "
                    f"{ast.unparse(node.generators[0].iter)[:40]}"
                )

    assert offenders == [], (
        f"unordered iteration that can reach output: {offenders}"
    )


@pytest.mark.parametrize("seed_a,seed_b", [("0", "1"), ("3", "77")])
def test_the_seeded_generators_are_byte_reproducible(seed_a, seed_b):
    """Synthetic data that varies between processes makes every seeded run a fiction.

    Checked across processes rather than twice in one, because a single process
    fixes the hash seed and the whole class of bug is invisible there.
    """
    import hashlib
    import os
    import subprocess
    import sys

    # The generator pulls in numpy transitively. Skipping rather than failing when
    # it is absent keeps this honest in a minimal environment, and it still runs on
    # CI where the dependency is installed.
    pytest.importorskip("numpy")

    program = (
        "from my_jev.fleet_synth import generate_fleet_records;"
        "print(''.join(r.model_dump_json() for r in generate_fleet_records(16, seed=51)))"
    )
    digests = set()
    for seed in (seed_a, seed_b):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, check=True, env=env,
        )
        digests.add(hashlib.sha256(result.stdout.encode()).hexdigest())

    assert len(digests) == 1, "generate_fleet_records is not byte-reproducible"
