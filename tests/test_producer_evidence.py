from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from my_jev.producer_evidence import (
    MANIFEST_SCHEMA,
    SIGNATURE_DOMAIN,
    ProducerEvidenceManifest,
    build_producer_evidence_manifest,
    deterministic_manifest_bytes,
    sign_producer_evidence_manifest,
    verify_producer_evidence_manifest_signature,
)
from my_jev.uhp_signature import public_key_id


AUTHORITY = {
    "dispatch_allowed": False,
    "approval_granted": False,
    "claim_acquired": False,
    "mutation_allowed": False,
    "routing_authority_changed": False,
}


def manifest() -> ProducerEvidenceManifest:
    key = Ed25519PrivateKey.generate()
    return build_producer_evidence_manifest(
        response_id="resp_provenance_test",
        receipt_id="receipt-provenance-test",
        consumer_session_id="pi-provenance-test",
        project_fingerprint="a" * 64,
        snapshot_sha256="b" * 64,
        stored_response_sha256="c" * 64,
        stored_response_signature_sha256="d" * 64,
        producer_key_id=public_key_id(key.public_key()),
        source_snapshot_raw_sha256="e" * 64,
        source_snapshot_canonical_sha256="f" * 64,
        recommendation_sha256="1" * 64,
        trace_sha256="2" * 64,
        systemone_config_sha256="3" * 64,
        my_jev_head="4" * 40,
        harnessrouter_head="5" * 40,
        harnessrouter_driver_git_blob_sha1="6" * 40,
        systemone_provider_git_blob_sha1="7" * 40,
        systemone_package_manifest_sha256="8" * 64,
        producer_python={
            "executable_sha256": "9" * 64,
            "isolated": True,
            "ignore_environment": True,
            "no_site": True,
        },
        heartbeat_mcp_python={
            "executable_sha256": "a" * 64,
            "isolated": True,
            "ignore_environment": True,
            "no_site": True,
        },
        removed_environment_keys=["PYTHONPATH", "TYPESAFE_API_KEY"],
        compiled_at="2026-09-25T14:00:00Z",
        receipt_expires_at="2026-09-25T14:05:00Z",
        authority=AUTHORITY,
    )


def test_manifest_serialization_is_exact_and_deterministic():
    value = manifest()
    first = deterministic_manifest_bytes(value)
    second = deterministic_manifest_bytes(
        json.loads(first.decode("utf-8"))
    )

    assert first == second
    assert first.endswith(b"\n")
    assert b"\n" not in first[:-1]
    parsed = json.loads(first)
    assert parsed["schema"] == MANIFEST_SCHEMA
    assert parsed["sanitized_environment"]["removed_keys"] == [
        "PYTHONPATH",
        "TYPESAFE_API_KEY",
    ]


def test_manifest_rejects_extra_fields_and_authority_widening():
    payload = manifest().model_dump(mode="json")
    payload["unexpected"] = False
    with pytest.raises(ValidationError):
        ProducerEvidenceManifest.model_validate(payload)

    payload = manifest().model_dump(mode="json")
    payload["authority"]["mutation_allowed"] = True
    with pytest.raises(ValidationError):
        ProducerEvidenceManifest.model_validate(payload)


def test_manifest_signature_uses_distinct_domain_and_exact_bytes():
    key = Ed25519PrivateKey.generate()
    value = manifest().model_copy(update={"producer_key_id": public_key_id(key.public_key())})
    raw = deterministic_manifest_bytes(value)
    envelope = sign_producer_evidence_manifest(raw, private_key=key)

    assert envelope.domain == SIGNATURE_DOMAIN
    assert envelope.key_id == value.producer_key_id

    verified = verify_producer_evidence_manifest_signature(
        raw,
        envelope,
        public_key=key.public_key(),
    )
    assert verified.manifest_sha256 == envelope.manifest_sha256


def test_manifest_signature_rejects_wrong_key_and_byte_tamper():
    signer = Ed25519PrivateKey.generate()
    wrong = Ed25519PrivateKey.generate()
    value = manifest().model_copy(
        update={"producer_key_id": public_key_id(signer.public_key())}
    )
    raw = deterministic_manifest_bytes(value)
    envelope = sign_producer_evidence_manifest(raw, private_key=signer)

    with pytest.raises(ValueError, match="key id"):
        verify_producer_evidence_manifest_signature(
            raw,
            envelope,
            public_key=wrong.public_key(),
        )

    tampered = raw.replace(b"resp_provenance_test", b"resp_provenance_tamper", 1)
    with pytest.raises(ValueError, match="manifest hash"):
        verify_producer_evidence_manifest_signature(
            tampered,
            envelope,
            public_key=signer.public_key(),
        )
