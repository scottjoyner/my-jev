from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .heartbeat_compile import TerminalRecommendation, compile_heartbeat_recommendation
from .heartbeat_snapshot import HeartbeatSnapshot, snapshot_sha256
from .uhp_advisory import SystemOneProvenance, build_uhp_response_fixture, canonical_sha256

PINNED_HARNESSROUTER_HEAD = "250de65d6e690abdef40e39d21591b4a807984a3"
PINNED_HARNESSROUTER_DRIVER_BLOB_SHA1 = "7cb3516a14b4f947e396a20735db4eb419a3db12"
PINNED_SYSTEMONE_PROVIDER_BLOB_SHA1 = "008ddd09fe8e2c85ee3b8316cf25062c28b59c1c"
PINNED_SYSTEMONE_PACKAGE_MANIFEST_SHA256 = "3a69281583ccccefd3e4d5422939703c842b00b92e2bfa9fc374293a94e15a74"
PINNED_SYSTEMONE_CONFIG_SHA256 = "459cc500b481878aa1445a6176bb8a6b61db51981696afcc6dd65f9fe3700f4e"
SCRIPT_MODEL = "script/s1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(framed).hexdigest()


def _git_head(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"could not resolve git head for {repo}: {proc.stderr.strip()}")
    head = proc.stdout.strip()
    if len(head) != 40:
        raise RuntimeError(f"unexpected git head for {repo}: {head!r}")
    return head


def _require_clean_git_checkout(repo: Path, label: str) -> str:
    head = _git_head(repo)
    proc = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} git status failed: {(proc.stderr or proc.stdout).strip()}"
        )
    if proc.stdout.strip():
        raise RuntimeError(
            f"{label} checkout must be clean for exact-head evidence:\n{proc.stdout}"
        )
    return head


def _systemone_probe_info(python: str) -> dict[str, str]:
    code = """
import hashlib, json, pathlib, systemone_harness
root = pathlib.Path(systemone_harness.__file__).resolve().parent
provider = root / "provider.py"
data = provider.read_bytes()
blob = b"blob " + str(len(data)).encode("ascii") + b"\\0" + data
manifest = []
for path in sorted(root.rglob("*.py")):
    rel = "systemone_harness/" + path.relative_to(root).as_posix()
    raw = path.read_bytes()
    framed = b"blob " + str(len(raw)).encode("ascii") + b"\\0" + raw
    manifest.append(rel + "\\t" + hashlib.sha1(framed).hexdigest())
manifest_text = "\\n".join(manifest) + "\\n"
print(json.dumps({
    "module": str(pathlib.Path(systemone_harness.__file__).resolve()),
    "provider_sha256": hashlib.sha256(data).hexdigest(),
    "provider_git_blob_sha1": hashlib.sha1(blob).hexdigest(),
    "package_manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
    "package_python_file_count": len(manifest),
}))
"""
    proc = subprocess.run(
        [python, "-c", code],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "HarnessRouter Python cannot import systemone_harness; use the runner "
            f"environment or pass --harnessrouter-python: {proc.stderr.strip()}"
        )
    try:
        payload = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("could not read SystemOneHarness probe metadata") from exc
    provider_sha = str(payload.get("provider_sha256") or "")
    if len(provider_sha) != 64:
        raise RuntimeError("SystemOneHarness provider.py hash was not a SHA-256")
    provider_blob = str(payload.get("provider_git_blob_sha1") or "")
    if provider_blob != PINNED_SYSTEMONE_PROVIDER_BLOB_SHA1:
        raise RuntimeError(
            "SystemOneHarness provider.py does not match the reviewed upstream blob: "
            f"expected {PINNED_SYSTEMONE_PROVIDER_BLOB_SHA1}, got {provider_blob or '<missing>'}"
        )
    package_manifest = str(payload.get("package_manifest_sha256") or "")
    if package_manifest != PINNED_SYSTEMONE_PACKAGE_MANIFEST_SHA256:
        raise RuntimeError(
            "installed SystemOneHarness Python package does not match the reviewed tree: "
            f"expected {PINNED_SYSTEMONE_PACKAGE_MANIFEST_SHA256}, "
            f"got {package_manifest or '<missing>'}"
        )
    return {
        "module": str(payload.get("module") or ""),
        "provider_sha256": provider_sha,
        "provider_git_blob_sha1": provider_blob,
        "package_manifest_sha256": package_manifest,
        "package_python_file_count": int(payload.get("package_python_file_count") or 0),
    }


def _safe_script_value(value: str) -> str:
    if any(char in value for char in (",", "(", ")", "=")):
        raise ValueError(
            "HarnessRouter ScriptProvider probe values may not contain ',', '(', ')' or '='"
        )
    return value


def _probe_choices(snapshot: HeartbeatSnapshot) -> tuple[str, str, str]:
    mode = "act"
    fleet = snapshot.fleet.eligible_handles[0] if snapshot.fleet.eligible_handles else "none"
    context = snapshot.knowledge.note_refs[0] if snapshot.knowledge.note_refs else "none"
    return mode, _safe_script_value(fleet), _safe_script_value(context)


def _script_entry(snapshot: HeartbeatSnapshot) -> str:
    mode, fleet, context = _probe_choices(snapshot)
    return (
        "recommend("
        f"mode={mode},"
        f"fleet_handle={fleet},"
        f"context_focus={context}"
        ")"
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_launcher(
    *,
    package_root: Path,
    snapshot: Path,
    my_jev_python: str,
    source_root: Path,
) -> Path:
    package_root.mkdir(parents=True, exist_ok=True)
    config_source = _repo_root() / "configs" / "systemone" / "hermes-heartbeat-advisory.yaml"
    if not config_source.is_file():
        raise RuntimeError(f"missing checked-in System-One config: {config_source}")
    (package_root / "config.yaml").write_bytes(config_source.read_bytes())

    launcher = package_root / "run-heartbeat-mcp.sh"
    python_q = shlex.quote(str(Path(my_jev_python).resolve()))
    snapshot_q = shlex.quote(str(snapshot))
    root_q = shlex.quote(str(package_root))
    source_q = shlex.quote(str(source_root))
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"export PLUGIN_ROOT={root_q}\n"
        f"export PYTHONPATH={source_q}\n"
        f"exec {python_q} -m my_jev.heartbeat_mcp --snapshot {snapshot_q}\n",
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    return launcher


def _read_ndjson(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"HarnessRouter emitted non-JSON output: {line[:300]}") from exc
        if isinstance(value, dict):
            events.append(value)
    return events


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _recommend_step(trace: dict[str, Any]) -> dict[str, Any]:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise RuntimeError("HarnessRouter trace has no steps list")
    matches = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("action") == "recommend"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one recommend step, got {len(matches)}")
    return matches[0]


def _confidence(step: dict[str, Any]) -> float:
    raw = step.get("action_confidence", step.get("weakest"))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("recommend step did not record action confidence") from exc
    if not (0.0 <= value <= 1.0):
        raise RuntimeError("recommend confidence is outside [0,1]")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned HarnessRouter System-One scripted-provider probe over a fresh "
            "heartbeat snapshot, then compile its recommendation into a bound UHP response."
        )
    )
    parser.add_argument("--harnessrouter-repo", type=Path, required=True)
    parser.add_argument("--harnessrouter-python", default=sys.executable)
    parser.add_argument(
        "--expected-harnessrouter-head",
        default=PINNED_HARNESSROUTER_HEAD,
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--consumer-session-id", required=True)
    parser.add_argument("--project-cwd", type=Path, required=True)
    parser.add_argument("--receipt-id", default="harnessrouter-script-probe")
    parser.add_argument("--response-id", default="resp_harnessrouter_script_probe")
    parser.add_argument("--uhp-session-id", default="hsess-harnessrouter-script-probe")
    parser.add_argument("--harness-id", default="chrn_system_one")
    parser.add_argument("--ttl-seconds", type=int, default=600)
    parser.add_argument("--task-focus")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    harnessrouter_repo = args.harnessrouter_repo.resolve()
    driver = harnessrouter_repo / "runner" / "systemone_driver.py"
    if not driver.is_file():
        raise RuntimeError(f"HarnessRouter System-One driver not found: {driver}")
    my_jev_root = _repo_root()
    my_jev_head_before = _require_clean_git_checkout(my_jev_root, "my-jev")
    actual_hr_head = _require_clean_git_checkout(harnessrouter_repo, "HarnessRouter")
    if actual_hr_head != args.expected_harnessrouter_head:
        raise RuntimeError(
            "HarnessRouter exact-head mismatch: "
            f"expected {args.expected_harnessrouter_head}, got {actual_hr_head}"
        )
    driver_blob = _git_blob_sha1(driver)
    if driver_blob != PINNED_HARNESSROUTER_DRIVER_BLOB_SHA1:
        raise RuntimeError(
            "HarnessRouter systemone_driver.py does not match the reviewed blob: "
            f"expected {PINNED_HARNESSROUTER_DRIVER_BLOB_SHA1}, got {driver_blob}"
        )

    snapshot_path = args.snapshot.resolve()
    snapshot = HeartbeatSnapshot.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )
    source_snapshot_sha = snapshot_sha256(snapshot)
    project_cwd = args.project_cwd.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    workspace = output_dir / "workspace"
    package_root = output_dir / "package"
    response_path = output_dir / "stored-uhp-response.json"
    snapshot_evidence_path = output_dir / "source-heartbeat-snapshot.json"
    recommendation_path = workspace / "hermes-system-one-recommendation.json"
    trace_path = workspace / "trace.json"
    report_path = output_dir / "harnessrouter-probe-evidence.json"

    for evidence_path in (
        recommendation_path,
        trace_path,
        response_path,
        snapshot_evidence_path,
        report_path,
    ):
        if evidence_path.exists():
            raise RuntimeError(
                f"refusing to overwrite existing probe evidence: {evidence_path}; "
                "use a fresh --output-dir"
            )
    workspace.mkdir(parents=True, exist_ok=True)
    _atomic_json(snapshot_evidence_path, snapshot.model_dump(mode="json"))

    config_source = _repo_root() / "configs" / "systemone" / "hermes-heartbeat-advisory.yaml"
    config_sha256 = _sha256_file(config_source)
    if config_sha256 != PINNED_SYSTEMONE_CONFIG_SHA256:
        raise RuntimeError(
            "System-One config does not match the reviewed bytes: "
            f"expected {PINNED_SYSTEMONE_CONFIG_SHA256}, got {config_sha256}"
        )
    launcher = _write_launcher(
        package_root=package_root,
        snapshot=snapshot_path,
        my_jev_python=sys.executable,
        source_root=_repo_root() / "src",
    )
    systemone_info = _systemone_probe_info(args.harnessrouter_python)
    script_entry = _script_entry(snapshot)

    job = {
        "cwd": str(workspace),
        "model": "jev-script-probe-unused",
        "prompt": "Produce exactly one bounded advisory recommendation from the heartbeat snapshot.",
        "provider": "typesafe",
        "base_url": "http://127.0.0.1:1/v1",
        "api_key": "script-probe-no-network",
        "mcp_servers": [
            {
                "name": "hermes-heartbeat",
                "command": str(launcher),
                "args": [],
            }
        ],
        "tools_disabled": [],
        "max_turns": 2,
        "metadata": {"systemone": {"script": [script_entry]}},
    }

    proc = subprocess.run(
        [args.harnessrouter_python, str(driver), json.dumps(job, separators=(",", ":"))],
        check=False,
        text=True,
        capture_output=True,
    )
    events = _read_ndjson(proc.stdout)
    result_events = [event for event in events if event.get("type") == "result"]
    if proc.returncode != 0 or len(result_events) != 1:
        raise RuntimeError(
            "HarnessRouter scripted System-One probe failed: "
            f"exit={proc.returncode}, results={len(result_events)}, "
            f"stderr={proc.stderr.strip()}"
        )
    result = result_events[0]
    if result.get("is_error") is True:
        raise RuntimeError(f"HarnessRouter result is an error: {result}")
    if result.get("model") != SCRIPT_MODEL:
        raise RuntimeError(
            f"expected scripted provider model {SCRIPT_MODEL!r}, got {result.get('model')!r}"
        )
    if not recommendation_path.is_file():
        raise RuntimeError("HarnessRouter did not persist the recommendation artifact")
    if not trace_path.is_file():
        raise RuntimeError("HarnessRouter did not persist trace.json")

    recommendation = TerminalRecommendation.model_validate_json(
        recommendation_path.read_text(encoding="utf-8")
    )
    if recommendation.snapshot_sha256 != source_snapshot_sha:
        raise RuntimeError("recommendation is not bound to the source heartbeat snapshot")

    expected_mode, expected_fleet, expected_context = _probe_choices(snapshot)
    actual_fleet = recommendation.advice.fleet_handle or "none"
    actual_context = recommendation.advice.context_focus or "none"
    if (
        recommendation.advice.mode != expected_mode
        or actual_fleet != expected_fleet
        or actual_context != expected_context
    ):
        raise RuntimeError(
            "HarnessRouter recommendation does not match the deterministic scripted choices"
        )

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace.get("config_version") != 1:
        raise RuntimeError(
            f"expected checked-in System-One config version 1, got {trace.get('config_version')!r}"
        )
    step = _recommend_step(trace)
    if step.get("verdict") != "run":
        raise RuntimeError(f"scripted recommend action did not run: {step.get('verdict')!r}")
    mode_confidence = _confidence(step)
    trace_sha = _sha256_file(trace_path)

    compiled_at = datetime.now(UTC).replace(microsecond=0)
    provenance = SystemOneProvenance(
        system_one_config_version=str(trace["config_version"]),
        model_revision=(
            f"{SCRIPT_MODEL}:provider.py@{systemone_info['provider_sha256']}"
        ),
        knowledge_revision=snapshot.knowledge.knowledge_revision,
        neo4j_snapshot_id=snapshot.knowledge.neo4j_snapshot_id,
        fleet_projection_generation=snapshot.fleet.projection_generation,
        fleet_projection_checksum=snapshot.fleet.projection_checksum,
        trace_sha256=trace_sha,
    )
    profile = compile_heartbeat_recommendation(
        snapshot,
        recommendation,
        receipt_id=args.receipt_id,
        consumer_session_id=args.consumer_session_id,
        project_cwd=project_cwd,
        compiled_at=compiled_at,
        ttl_seconds=args.ttl_seconds,
        mode_confidence=mode_confidence,
        task_focus=args.task_focus,
        provenance=provenance,
    )
    response = build_uhp_response_fixture(
        profile,
        response_id=args.response_id,
        session_id=args.uhp_session_id,
        harness_id=args.harness_id,
        model=SCRIPT_MODEL,
        created_at=compiled_at,
    )
    _atomic_json(response_path, response)

    recommendation_authority = recommendation.authority.model_dump(mode="json")
    compiled_authority = profile.authority.model_dump(mode="json")
    assertions = {
        "harnessrouter_exact_head": actual_hr_head == args.expected_harnessrouter_head,
        "harnessrouter_driver_blob_pinned":
            driver_blob == PINNED_HARNESSROUTER_DRIVER_BLOB_SHA1,
        "systemone_provider_blob_pinned":
            systemone_info["provider_git_blob_sha1"] == PINNED_SYSTEMONE_PROVIDER_BLOB_SHA1,
        "systemone_package_manifest_pinned":
            systemone_info["package_manifest_sha256"]
            == PINNED_SYSTEMONE_PACKAGE_MANIFEST_SHA256,
        "systemone_config_pinned": config_sha256 == PINNED_SYSTEMONE_CONFIG_SHA256,
        "script_provider_used": result.get("model") == SCRIPT_MODEL,
        "single_recommend_step": step.get("action") == "recommend",
        "recommend_step_ran": step.get("verdict") == "run",
        "config_version_loaded": trace.get("config_version") == 1,
        "snapshot_hash_preserved": recommendation.snapshot_sha256 == source_snapshot_sha,
        "recommendation_evidence_only": recommendation.evidence_only is True,
        "recommendation_runtime_authority_unchanged":
            recommendation.runtime_authority_changed is False,
        "recommendation_authority_all_false":
            all(value is False for value in recommendation_authority.values()),
        "compiled_authority_all_false":
            all(value is False for value in compiled_authority.values()),
        "consumer_session_bound":
            profile.binding.consumer_session_id == args.consumer_session_id,
        "trace_hash_bound": profile.provenance.trace_sha256 == trace_sha,
        "served_model_is_script": response["model"] == SCRIPT_MODEL,
        "no_model_fallback": "model_fallback" not in response["metadata"],
    }
    my_jev_head_after = _require_clean_git_checkout(my_jev_root, "my-jev")
    harnessrouter_head_after = _require_clean_git_checkout(
        harnessrouter_repo,
        "HarnessRouter",
    )
    if my_jev_head_after != my_jev_head_before or harnessrouter_head_after != actual_hr_head:
        raise RuntimeError("producer source checkout HEAD changed during the probe")

    verdict = "pass" if all(assertions.values()) else "fail"
    evidence = {
        "schema": "my-jev-harnessrouter-script-probe-v1",
        "verdict": verdict,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "harnessrouter_head": actual_hr_head,
        "harnessrouter_driver_sha256": _sha256_file(driver),
        "harnessrouter_driver_git_blob_sha1": driver_blob,
        "systemone_config_sha256": config_sha256,
        "systemone_harness": systemone_info,
        "my_jev_head": my_jev_head_before,
        "source_checkouts_clean": True,
        "source_heads_stable": True,
        "source_snapshot": str(snapshot_path),
        "source_snapshot_evidence": str(snapshot_evidence_path),
        "source_snapshot_evidence_raw_sha256": _sha256_file(snapshot_evidence_path),
        "snapshot_sha256": source_snapshot_sha,
        "script_entry": script_entry,
        "result": result,
        "recommendation_sha256": _sha256_file(recommendation_path),
        "trace_sha256": trace_sha,
        "mode_confidence": mode_confidence,
        "profile_sha256": canonical_sha256(profile),
        "response_sha256": canonical_sha256(response),
        "stored_response_raw_sha256": _sha256_file(response_path),
        "stored_response": str(response_path),
        "consumer_session_id": profile.binding.consumer_session_id,
        "project_fingerprint": profile.binding.project_fingerprint,
        "receipt_id": profile.receipt_id,
        "response_id": response["id"],
        "assertions": assertions,
    }
    _atomic_json(report_path, evidence)
    print(json.dumps({**evidence, "report": str(report_path)}, sort_keys=True))
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
