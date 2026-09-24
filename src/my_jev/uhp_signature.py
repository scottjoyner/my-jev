from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .uhp_advisory import canonical_sha256

SIGNATURE_DOMAIN = "hermes-system-one-uhp-ed25519-v1"
SIGNATURE_METADATA_KEY = "hermes_system_one_signature"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def unsigned_response(response: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(response))
    metadata = dict(payload.get("metadata") or {})
    metadata.pop(SIGNATURE_METADATA_KEY, None)
    payload["metadata"] = metadata
    return payload


def signature_claims(response: Mapping[str, Any]) -> dict[str, str]:
    unsigned = unsigned_response(response)
    metadata = dict(unsigned.get("metadata") or {})
    profile = metadata.get("hermes_system_one")
    if not isinstance(profile, Mapping):
        raise ValueError("response metadata.hermes_system_one is required for signing")

    response_id = str(unsigned.get("id") or "").strip()
    uhp_session_id = str(metadata.get("session_id") or "").strip()
    harness_id = str(metadata.get("harness_id") or "").strip()
    served_model = str(unsigned.get("model") or "").strip()
    if not response_id or not uhp_session_id or not harness_id or not served_model:
        raise ValueError("response identity is incomplete")

    return {
        "domain": SIGNATURE_DOMAIN,
        "response_id": response_id,
        "uhp_session_id": uhp_session_id,
        "harness_id": harness_id,
        "served_model": served_model,
        "profile_sha256": canonical_sha256(profile),
        "unsigned_response_sha256": canonical_sha256(unsigned),
    }


def signature_preimage(response: Mapping[str, Any]) -> bytes:
    return _canonical_json(signature_claims(response)).encode("utf-8")


def public_key_id(public_key: Ed25519PublicKey) -> str:
    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return "ed25519:" + hashlib.sha256(der).hexdigest()


def load_private_key(path: str | Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(
        Path(path).read_bytes(),
        password=None,
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("signing key must be an Ed25519 private key")
    return key


def load_public_key(path: str | Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("verification key must be an Ed25519 public key")
    return key


def sign_uhp_response(
    response: Mapping[str, Any],
    *,
    private_key: Ed25519PrivateKey,
) -> dict[str, Any]:
    payload = unsigned_response(response)
    claims = signature_claims(payload)
    preimage = _canonical_json(claims).encode("utf-8")
    signature = private_key.sign(preimage)
    envelope = {
        "scheme": "ed25519",
        "domain": SIGNATURE_DOMAIN,
        "key_id": public_key_id(private_key.public_key()),
        "profile_sha256": claims["profile_sha256"],
        "unsigned_response_sha256": claims["unsigned_response_sha256"],
        "preimage_sha256": hashlib.sha256(preimage).hexdigest(),
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    metadata = dict(payload["metadata"])
    metadata[SIGNATURE_METADATA_KEY] = envelope
    payload["metadata"] = metadata
    return payload


def verify_uhp_response_signature(
    response: Mapping[str, Any],
    *,
    public_key: Ed25519PublicKey,
) -> dict[str, Any]:
    metadata = dict(response.get("metadata") or {})
    envelope = metadata.get(SIGNATURE_METADATA_KEY)
    if not isinstance(envelope, Mapping):
        raise ValueError("response has no Hermes System-One signature envelope")

    claims = signature_claims(response)
    preimage = _canonical_json(claims).encode("utf-8")
    expected_key_id = public_key_id(public_key)

    if envelope.get("scheme") != "ed25519":
        raise ValueError("unsupported signature scheme")
    if envelope.get("domain") != SIGNATURE_DOMAIN:
        raise ValueError("signature domain mismatch")
    if envelope.get("key_id") != expected_key_id:
        raise ValueError("signature key id mismatch")
    if envelope.get("profile_sha256") != claims["profile_sha256"]:
        raise ValueError("signed profile hash mismatch")
    if envelope.get("unsigned_response_sha256") != claims["unsigned_response_sha256"]:
        raise ValueError("signed response hash mismatch")
    if envelope.get("preimage_sha256") != hashlib.sha256(preimage).hexdigest():
        raise ValueError("signature preimage hash mismatch")

    try:
        signature = base64.b64decode(str(envelope.get("signature_b64") or ""), validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid base64 signature") from exc

    try:
        public_key.verify(signature, preimage)
    except InvalidSignature as exc:
        raise ValueError("invalid Ed25519 signature") from exc

    return dict(envelope)
