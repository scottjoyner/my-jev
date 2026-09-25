from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .uhp_signature import public_key_id

MANIFEST_SCHEMA = "hermes-system-one-producer-evidence-manifest-v1"
SIGNATURE_SCHEMA = "hermes-system-one-producer-evidence-signature-v1"
SIGNATURE_DOMAIN = "hermes-system-one-producer-evidence-manifest-ed25519-v1"

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SHA1_PATTERN = r"^[0-9a-f]{40}$"
_KEY_ID_PATTERN = r"^ed25519:[0-9a-f]{64}$"


class _ExactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProducerPythonEvidence(_ExactModel):
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    isolated: Literal[True]
    ignore_environment: Literal[True]
    no_site: Literal[True]


class SanitizedEnvironmentEvidence(_ExactModel):
    removed_keys: list[str]
    provider_credentials_present: Literal[False]
    startup_injection_present: Literal[False]

    @field_validator("removed_keys")
    @classmethod
    def _validate_removed_keys(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("removed_keys must be sorted and unique")
        for key in value:
            if not key or len(key) > 128:
                raise ValueError("removed environment key must be non-empty and bounded")
        return value


class ProducerAuthority(_ExactModel):
    dispatch_allowed: Literal[False]
    approval_granted: Literal[False]
    claim_acquired: Literal[False]
    mutation_allowed: Literal[False]
    routing_authority_changed: Literal[False]


class ProducerEvidenceManifest(_ExactModel):
    schema: Literal["hermes-system-one-producer-evidence-manifest-v1"] = MANIFEST_SCHEMA
    response_id: str = Field(pattern=r"^resp_[A-Za-z0-9._:-]+$", max_length=256)
    receipt_id: str = Field(min_length=1, max_length=200)
    consumer_session_id: str = Field(min_length=1, max_length=128)
    project_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)

    stored_response_sha256: str = Field(pattern=_SHA256_PATTERN)
    stored_response_signature_sha256: str = Field(pattern=_SHA256_PATTERN)
    producer_key_id: str = Field(pattern=_KEY_ID_PATTERN)

    source_snapshot_raw_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_snapshot_canonical_sha256: str = Field(pattern=_SHA256_PATTERN)
    recommendation_sha256: str = Field(pattern=_SHA256_PATTERN)
    trace_sha256: str = Field(pattern=_SHA256_PATTERN)
    systemone_config_sha256: str = Field(pattern=_SHA256_PATTERN)

    my_jev_head: str = Field(pattern=_SHA1_PATTERN)
    harnessrouter_head: str = Field(pattern=_SHA1_PATTERN)
    harnessrouter_driver_git_blob_sha1: str = Field(pattern=_SHA1_PATTERN)
    systemone_provider_git_blob_sha1: str = Field(pattern=_SHA1_PATTERN)
    systemone_package_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    producer_python: ProducerPythonEvidence
    heartbeat_mcp_python: ProducerPythonEvidence
    sanitized_environment: SanitizedEnvironmentEvidence

    compiled_at: str = Field(min_length=20, max_length=40)
    receipt_expires_at: str = Field(min_length=20, max_length=40)
    authority: ProducerAuthority


class ProducerEvidenceSignature(_ExactModel):
    schema: Literal["hermes-system-one-producer-evidence-signature-v1"] = SIGNATURE_SCHEMA
    scheme: Literal["ed25519"] = "ed25519"
    domain: Literal["hermes-system-one-producer-evidence-manifest-ed25519-v1"] = (
        SIGNATURE_DOMAIN
    )
    key_id: str = Field(pattern=_KEY_ID_PATTERN)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    preimage_sha256: str = Field(pattern=_SHA256_PATTERN)
    signature_b64: str = Field(min_length=1, max_length=128)


def deterministic_manifest_bytes(
    manifest: ProducerEvidenceManifest | Mapping[str, Any],
) -> bytes:
    value = (
        manifest.model_dump(mode="json")
        if isinstance(manifest, ProducerEvidenceManifest)
        else ProducerEvidenceManifest.model_validate(manifest).model_dump(mode="json")
    )
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def manifest_signature_preimage(manifest_bytes: bytes) -> bytes:
    return SIGNATURE_DOMAIN.encode("utf-8") + b"\0" + manifest_bytes


def sign_producer_evidence_manifest(
    manifest_bytes: bytes,
    *,
    private_key: Ed25519PrivateKey,
) -> ProducerEvidenceSignature:
    preimage = manifest_signature_preimage(manifest_bytes)
    return ProducerEvidenceSignature(
        key_id=public_key_id(private_key.public_key()),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        preimage_sha256=hashlib.sha256(preimage).hexdigest(),
        signature_b64=base64.b64encode(private_key.sign(preimage)).decode("ascii"),
    )


def verify_producer_evidence_manifest_signature(
    manifest_bytes: bytes,
    envelope: ProducerEvidenceSignature | Mapping[str, Any],
    *,
    public_key: Ed25519PublicKey,
) -> ProducerEvidenceSignature:
    signed = (
        envelope
        if isinstance(envelope, ProducerEvidenceSignature)
        else ProducerEvidenceSignature.model_validate(envelope)
    )
    if signed.key_id != public_key_id(public_key):
        raise ValueError("producer evidence signature key id mismatch")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if signed.manifest_sha256 != manifest_sha:
        raise ValueError("producer evidence manifest hash mismatch")

    preimage = manifest_signature_preimage(manifest_bytes)
    preimage_sha = hashlib.sha256(preimage).hexdigest()
    if signed.preimage_sha256 != preimage_sha:
        raise ValueError("producer evidence preimage hash mismatch")

    try:
        signature = base64.b64decode(signed.signature_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid producer evidence base64 signature") from exc
    if len(signature) != 64:
        raise ValueError("invalid producer evidence Ed25519 signature length")
    if base64.b64encode(signature).decode("ascii") != signed.signature_b64:
        raise ValueError("non-canonical producer evidence base64 signature")

    try:
        public_key.verify(signature, preimage)
    except InvalidSignature as exc:
        raise ValueError("invalid producer evidence Ed25519 signature") from exc
    return signed


def build_producer_evidence_manifest(
    *,
    response_id: str,
    receipt_id: str,
    consumer_session_id: str,
    project_fingerprint: str,
    snapshot_sha256: str,
    stored_response_sha256: str,
    stored_response_signature_sha256: str,
    producer_key_id: str,
    source_snapshot_raw_sha256: str,
    source_snapshot_canonical_sha256: str,
    recommendation_sha256: str,
    trace_sha256: str,
    systemone_config_sha256: str,
    my_jev_head: str,
    harnessrouter_head: str,
    harnessrouter_driver_git_blob_sha1: str,
    systemone_provider_git_blob_sha1: str,
    systemone_package_manifest_sha256: str,
    producer_python: Mapping[str, Any],
    heartbeat_mcp_python: Mapping[str, Any],
    removed_environment_keys: Sequence[str],
    compiled_at: str,
    receipt_expires_at: str,
    authority: Mapping[str, Any],
) -> ProducerEvidenceManifest:
    removed = sorted(set(str(key) for key in removed_environment_keys))
    return ProducerEvidenceManifest(
        response_id=response_id,
        receipt_id=receipt_id,
        consumer_session_id=consumer_session_id,
        project_fingerprint=project_fingerprint,
        snapshot_sha256=snapshot_sha256,
        stored_response_sha256=stored_response_sha256,
        stored_response_signature_sha256=stored_response_signature_sha256,
        producer_key_id=producer_key_id,
        source_snapshot_raw_sha256=source_snapshot_raw_sha256,
        source_snapshot_canonical_sha256=source_snapshot_canonical_sha256,
        recommendation_sha256=recommendation_sha256,
        trace_sha256=trace_sha256,
        systemone_config_sha256=systemone_config_sha256,
        my_jev_head=my_jev_head,
        harnessrouter_head=harnessrouter_head,
        harnessrouter_driver_git_blob_sha1=harnessrouter_driver_git_blob_sha1,
        systemone_provider_git_blob_sha1=systemone_provider_git_blob_sha1,
        systemone_package_manifest_sha256=systemone_package_manifest_sha256,
        producer_python=ProducerPythonEvidence.model_validate(producer_python),
        heartbeat_mcp_python=ProducerPythonEvidence.model_validate(
            heartbeat_mcp_python
        ),
        sanitized_environment=SanitizedEnvironmentEvidence(
            removed_keys=removed,
            provider_credentials_present=False,
            startup_injection_present=False,
        ),
        compiled_at=compiled_at,
        receipt_expires_at=receipt_expires_at,
        authority=ProducerAuthority.model_validate(authority),
    )
