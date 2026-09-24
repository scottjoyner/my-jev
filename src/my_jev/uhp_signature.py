from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

SIGNATURE_SCHEMA = "hermes-system-one-detached-signature-v1"
SIGNATURE_DOMAIN = "hermes-system-one-uhp-response-bytes-ed25519-v1"


def signature_preimage(response_bytes: bytes) -> bytes:
    return SIGNATURE_DOMAIN.encode("utf-8") + b"\0" + response_bytes


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


def sign_uhp_response_bytes(
    response_bytes: bytes,
    *,
    private_key: Ed25519PrivateKey,
) -> dict[str, Any]:
    preimage = signature_preimage(response_bytes)
    signature = private_key.sign(preimage)
    return {
        "schema": SIGNATURE_SCHEMA,
        "scheme": "ed25519",
        "domain": SIGNATURE_DOMAIN,
        "key_id": public_key_id(private_key.public_key()),
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "preimage_sha256": hashlib.sha256(preimage).hexdigest(),
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }


def verify_uhp_response_bytes(
    response_bytes: bytes,
    envelope: Mapping[str, Any],
    *,
    public_key: Ed25519PublicKey,
) -> dict[str, Any]:
    if envelope.get("schema") != SIGNATURE_SCHEMA:
        raise ValueError("signature schema mismatch")
    if envelope.get("scheme") != "ed25519":
        raise ValueError("unsupported signature scheme")
    if envelope.get("domain") != SIGNATURE_DOMAIN:
        raise ValueError("signature domain mismatch")
    if envelope.get("key_id") != public_key_id(public_key):
        raise ValueError("signature key id mismatch")

    response_sha = hashlib.sha256(response_bytes).hexdigest()
    if envelope.get("response_sha256") != response_sha:
        raise ValueError("signed response hash mismatch")

    preimage = signature_preimage(response_bytes)
    preimage_sha = hashlib.sha256(preimage).hexdigest()
    if envelope.get("preimage_sha256") != preimage_sha:
        raise ValueError("signature preimage hash mismatch")

    try:
        signature = base64.b64decode(str(envelope.get("signature_b64") or ""), validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid base64 signature") from exc
    if len(signature) != 64:
        raise ValueError("invalid Ed25519 signature length")

    try:
        public_key.verify(signature, preimage)
    except InvalidSignature as exc:
        raise ValueError("invalid Ed25519 signature") from exc

    return dict(envelope)
