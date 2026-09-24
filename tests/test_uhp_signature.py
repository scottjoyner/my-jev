from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from my_jev.uhp_signature import (
    SIGNATURE_DOMAIN,
    load_private_key,
    load_public_key,
    public_key_id,
    sign_uhp_response,
    signature_claims,
    unsigned_response,
    verify_uhp_response_signature,
)


def response():
    return {
        "id": "resp_signature_test",
        "object": "response",
        "status": "completed",
        "model": "script/s1",
        "metadata": {
            "session_id": "hsess-signature-test",
            "harness_id": "chrn_system_one",
            "hermes_system_one": {
                "profile": "hermes-system-one-heartbeat-v1",
                "receipt_id": "receipt-signature-test",
                "binding": {
                    "consumer": "local-studio",
                    "consumer_session_id": "pi-signature-test",
                    "project_fingerprint": "a" * 64,
                    "snapshot_sha256": "b" * 64,
                },
                "authority": {
                    "dispatch_allowed": False,
                    "approval_granted": False,
                    "claim_acquired": False,
                    "mutation_allowed": False,
                    "routing_authority_changed": False,
                },
            },
        },
    }


def test_ed25519_signature_binds_profile_and_unsigned_response():
    key = Ed25519PrivateKey.generate()
    signed = sign_uhp_response(response(), private_key=key)

    envelope = verify_uhp_response_signature(signed, public_key=key.public_key())
    claims = signature_claims(signed)

    assert envelope["scheme"] == "ed25519"
    assert envelope["domain"] == SIGNATURE_DOMAIN
    assert envelope["key_id"] == public_key_id(key.public_key())
    assert envelope["profile_sha256"] == claims["profile_sha256"]
    assert envelope["unsigned_response_sha256"] == claims["unsigned_response_sha256"]
    assert "hermes_system_one_signature" not in unsigned_response(signed)["metadata"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other-model"),
        ("id", "resp_other"),
    ],
)
def test_signature_rejects_response_identity_tampering(field, value):
    key = Ed25519PrivateKey.generate()
    signed = sign_uhp_response(response(), private_key=key)
    signed[field] = value

    with pytest.raises(ValueError, match="signature|hash"):
        verify_uhp_response_signature(signed, public_key=key.public_key())


def test_signature_rejects_profile_tampering():
    key = Ed25519PrivateKey.generate()
    signed = sign_uhp_response(response(), private_key=key)
    tampered = deepcopy(signed)
    tampered["metadata"]["hermes_system_one"]["receipt_id"] = "receipt-tampered"

    with pytest.raises(ValueError, match="hash|signature"):
        verify_uhp_response_signature(tampered, public_key=key.public_key())


def test_signature_rejects_wrong_public_key():
    signer = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    signed = sign_uhp_response(response(), private_key=signer)

    with pytest.raises(ValueError, match="key id"):
        verify_uhp_response_signature(signed, public_key=other.public_key())


def test_pem_key_loading_round_trip(tmp_path):
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "producer-private.pem"
    public_path = tmp_path / "producer-public.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    loaded_private = load_private_key(private_path)
    loaded_public = load_public_key(public_path)
    signed = sign_uhp_response(response(), private_key=loaded_private)

    envelope = verify_uhp_response_signature(signed, public_key=loaded_public)
    assert envelope["key_id"] == public_key_id(private.public_key())
