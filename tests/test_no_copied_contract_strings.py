"""Published identifiers must be defined once and referenced, not copied.

A string that names a wire contract is not an implementation detail. This project
has already been bitten by exactly this class of problem -- `CONTRACT_SHA256`
moving while documentation kept the old value -- and the difference there was
caught by a deliberate test. The same reasoning applies to every identifier that
appears in more than one module.

Each test below names the copies that used to exist, because a guard that only says
"no duplicates" gives an operator nothing to go on when it fires.
"""

from __future__ import annotations

import ast
import pathlib
import re
from fnmatch import fnmatch

import pytest

from my_jev.heartbeat_compile import RECOMMENDATION_SCHEMA
from my_jev.gpu_plan import STAGE_TIMING_FILENAME, STAGE_TIMING_SUFFIX

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "my_jev"


def _modules_matching(pattern: str) -> dict[str, list[int]]:
    """Files containing the literal, and the line numbers, for a failure message."""
    hits: dict[str, list[int]] = {}
    for path in sorted(SRC.glob("*.py")):
        lines = [
            number
            for number, line in enumerate(path.read_text().splitlines(), start=1)
            if re.search(pattern, line)
        ]
        if lines:
            hits[path.name] = lines
    return hits


def test_the_recommendation_schema_is_written_once_and_referenced():
    """It was a bare literal in three modules while the constant sat unused.

    `heartbeat_compile` held the constant, its own `Literal[...]` annotation could
    not reference it, and `heartbeat_mcp` had a third copy inline. So the constant
    was dead and changing the published schema name would have changed one place
    out of three -- a heartbeat declaring a schema version its own writer no longer
    emits.
    """
    hits = _modules_matching(r'"hermes-system-one-recommendation-v1"')
    # Two remain, both in the defining module: the constant's own definition, and
    # the `Literal` annotation, which pydantic requires to be a literal. Every
    # other module must reference the constant.
    assert hits == {"heartbeat_compile.py": [30, 55]}, hits


def test_the_annotation_and_the_constant_are_the_same_string():
    """The `Literal` that cannot reference the constant must still agree with it."""
    source = (SRC / "heartbeat_compile.py").read_text()
    literal = re.search(r'Literal\["([^"]+)"\]', source)
    assert literal is not None
    assert literal.group(1) == RECOMMENDATION_SCHEMA


def test_the_timing_filename_pattern_is_written_once():
    """It was a literal in four places: the constant, the writer, the reader's glob,
    and the suffix stripped to recover a stage name."""
    hits = _modules_matching(r"\.timing\.json")
    # Only the constant's own definition and a prose comment may mention it.
    for name, lines in hits.items():
        source = (SRC / name).read_text().splitlines()
        for number in lines:
            line = source[number - 1]
            assert "STAGE_TIMING_FILENAME" in line or line.lstrip().startswith("#"), (
                f"{name}:{number} mentions the timing filename literally: {line!r}"
            )


def test_the_suffix_is_derived_from_the_pattern_not_written_out():
    """So renaming the file cannot leave the reader stripping the wrong suffix."""
    assert STAGE_TIMING_SUFFIX == STAGE_TIMING_FILENAME.format(name="")
    assert STAGE_TIMING_SUFFIX == ".timing.json"


def test_the_writer_emits_exactly_the_name_the_reader_globs_for():
    """The property the two derived values exist to guarantee."""
    written = STAGE_TIMING_FILENAME.format(name="train")
    assert written == "train.timing.json"
    assert written.endswith(STAGE_TIMING_SUFFIX)
    # And the reader's glob matches what the writer produced.
    assert fnmatch(written, f"*{STAGE_TIMING_SUFFIX}")
    assert written.removesuffix(STAGE_TIMING_SUFFIX) == "train"


@pytest.mark.parametrize(
    "identifier",
    [
        "hermes-system-one-heartbeat-v1",
        "benchmark-qualification-report-v1",
        "my-jev-market-evidence-v1",
    ],
)
def test_published_schema_identifiers_are_not_invented_twice(identifier):
    """A guard rather than a policy: it reports where the copies are.

    More than one module may legitimately mention a published name -- a
    `Literal` annotation plus the constant that documents it -- so this asserts a
    ceiling rather than a count, and the message names every location.
    """
    hits = _modules_matching(re.escape(identifier))
    assert len(hits) <= 2, f"{identifier} appears in {len(hits)} modules: {hits}"


def test_every_module_level_constant_is_referenced_somewhere():
    """The heuristic that started this, kept as a test.

    A constant defined and never referenced is either dead or a signal that the
    code that should use it was written against a literal instead. Both outcomes
    are worth a human reading the failure.
    """
    package = "\n".join(path.read_text() for path in sorted(SRC.glob("*.py")))
    tests = "\n".join(
        path.read_text() for path in sorted((SRC.parent.parent / "tests").glob("*.py"))
    )
    dead: dict[str, str] = {}
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            for target in targets:
                if not target.id.isupper():
                    continue
                occurrences = len(
                    re.findall(rf"\b{re.escape(target.id)}\b", package + tests)
                )
                if occurrences <= 1:
                    dead[f"{path.name}:{target.id}"] = target.id
    assert dead == {}, f"constants defined and never referenced: {sorted(dead)}"
