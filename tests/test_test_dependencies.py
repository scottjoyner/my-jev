"""No test may depend on a package the test install does not provide.

Found the hard way. `tests/test_server_surface.py` guarded itself with
`pytest.importorskip("fastapi")`, and CI installs only `.[dev]` — which did not list
`fastapi`. So the 29 tests covering the HTTP surface skipped on every CI run, silently
and successfully, and the build stayed green.

That is the failure mode worth guarding against rather than fixing once: a skipped test
reads as coverage, so a dependency that goes missing in CI removes the tests without
removing the green tick. Nothing fails; the suite simply stops checking something and
nobody finds out until the thing breaks in production.

The rule enforced here: every `pytest.importorskip` in the suite must name a module
that the *base* dependencies or the `dev` extra actually provide. Anything else has to
either be declared, or be deliberately marked as an optional-extra test whose skip is
expected and recorded.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

#: Extras that CI installs. Kept in step with `.github/workflows/ci.yml`, which runs
#: `pip install -e ".[dev]"` and nothing else. Adding an extra here without adding it
#: to that workflow would make this test lie, so the workflow is parsed too.
CI_INSTALLS = "dev"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _requirement_names(requirement: str) -> str:
    """`fastapi>=0.115` -> `fastapi`; handles extras and markers loosely."""
    name = re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def _provided_by_install() -> set[str]:
    data = _pyproject()
    project = data["project"]
    names = {_requirement_names(r) for r in project.get("dependencies", [])}
    for extra in (CI_INSTALLS,):
        names |= {_requirement_names(r) for r in project["optional-dependencies"][extra]}
    return names


def _ci_workflow_installs() -> str:
    return (ROOT / ".github" / "workflows" / "ci.yml").read_text()


def _importorskip_modules() -> dict[str, list[str]]:
    """Every module name passed to `pytest.importorskip`, and the files using it."""
    found: dict[str, list[str]] = {}
    for path in sorted(TESTS.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name != "importorskip":
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.setdefault(first.value, []).append(path.name)
    return found


def test_the_ci_workflow_installs_the_extra_this_file_assumes():
    """If the workflow stops installing `dev`, every claim below is void.

    Asserted first so a change to the workflow fails here rather than making the rest
    of this file quietly meaningless.
    """
    workflow = _ci_workflow_installs()
    assert f'pip install -e ".[{CI_INSTALLS}]"' in workflow, (
        f"CI no longer installs the {CI_INSTALLS!r} extra this file reasons about; "
        "update CI_INSTALLS to match, deliberately"
    )


def test_the_suite_actually_uses_importorskip_somewhere():
    """Otherwise the checks below pass by finding nothing.

    Added after an earlier version of a similar guard here found an empty subject list
    and reported success. A guard whose subjects come from the thing it guards cannot
    notice that thing disappearing.
    """
    assert _importorskip_modules(), "no importorskip found; is this file still needed?"


def test_every_importorskip_names_a_package_the_test_install_provides():
    """A skip that CI always takes is a test that CI never runs.

    `importorskip` is the right tool for a genuinely optional dependency. It is the
    wrong tool for one the test suite claims to need: that converts a missing
    declaration into a silently reduced test run.
    """
    provided = _provided_by_install()
    offenders = {
        module: files
        for module, files in _importorskip_modules().items()
        if module.lower().replace("_", "-") not in provided
    }
    assert offenders == {}, (
        "these tests skip unless an undeclared package is installed, so CI does not "
        f"run them: {offenders}. Declare the package in `dependencies` or in the "
        f"{CI_INSTALLS!r} extra, or move the test behind an explicit marker."
    )


@pytest.mark.parametrize("module", sorted(_importorskip_modules()))
def test_each_skipped_module_is_importable_here(module):
    """If it is declared, it should be installed -- so the skip should not fire.

    Catches the opposite failure: declaring a dependency and then still skipping on it
    in the environment where it exists, which hides a genuine import error behind a
    skip.
    """
    pytest.importorskip(module)


def test_no_test_file_is_skipped_whole():
    """`pytestmark = pytest.mark.skip` is as invisible as an importorskip."""
    offenders = []
    for path in sorted(TESTS.glob("*.py")):
        text = path.read_text()
        if re.search(r"^pytestmark\s*=\s*pytest\.mark\.skip", text, re.M):
            offenders.append(path.name)
        if re.search(r"^pytest\.mark\.skip\s*$", text, re.M):
            offenders.append(path.name)
    assert offenders == [], f"whole test files disabled unconditionally: {offenders}"


def test_the_skipping_imports_are_the_expected_ones():
    """A hard list, so a new skip has to be a decision rather than a reflex.

    This is the same discipline as the pinned `_CORRECTION_FIELDS` list and the pinned
    training defaults: a contract written down somewhere other than where it is
    implemented, or it is not a contract.
    """
    assert set(_importorskip_modules()) == {
        "fastapi",
        "httpx",
        "numpy",
        "torch",
    }, (
        "the set of optionally-skipped packages changed; if a new one is genuinely "
        "optional, declare why here rather than letting a skip appear on its own"
    )