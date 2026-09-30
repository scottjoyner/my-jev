import json
import os

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from my_jev.uhp_signature import (
    SIGNATURE_DOMAIN,
    load_private_key,
    load_public_key,
    public_key_id,
    sign_uhp_response_bytes,
    signature_preimage,
    verify_uhp_response_bytes,
)


def response_bytes():
    payload = {
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
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def test_ed25519_signature_binds_exact_stored_response_bytes():
    key = Ed25519PrivateKey.generate()
    raw = response_bytes()

    envelope = sign_uhp_response_bytes(raw, private_key=key)
    verified = verify_uhp_response_bytes(raw, envelope, public_key=key.public_key())

    assert verified["schema"] == "hermes-system-one-detached-signature-v1"
    assert verified["scheme"] == "ed25519"
    assert verified["domain"] == SIGNATURE_DOMAIN
    assert verified["key_id"] == public_key_id(key.public_key())
    assert len(signature_preimage(raw)) > len(raw)


def test_signature_rejects_one_byte_response_tampering():
    key = Ed25519PrivateKey.generate()
    raw = response_bytes()
    envelope = sign_uhp_response_bytes(raw, private_key=key)
    tampered = raw.replace(b"script/s1", b"script/s2", 1)

    with pytest.raises(ValueError, match="hash|signature"):
        verify_uhp_response_bytes(tampered, envelope, public_key=key.public_key())


def test_signature_rejects_wrong_public_key():
    signer = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    raw = response_bytes()
    envelope = sign_uhp_response_bytes(raw, private_key=signer)

    with pytest.raises(ValueError, match="key id"):
        verify_uhp_response_bytes(raw, envelope, public_key=other.public_key())


def test_signature_rejects_extra_envelope_fields():
    key = Ed25519PrivateKey.generate()
    raw = response_bytes()
    envelope = sign_uhp_response_bytes(raw, private_key=key)
    envelope["unexpected"] = False

    with pytest.raises(ValueError, match="shape"):
        verify_uhp_response_bytes(raw, envelope, public_key=key.public_key())


def test_signature_rejects_noncanonical_base64():
    key = Ed25519PrivateKey.generate()
    raw = response_bytes()
    envelope = sign_uhp_response_bytes(raw, private_key=key)
    # An extra '=' is decodable by permissive base64 readers but is not the
    # canonical representation emitted by the signer.
    envelope["signature_b64"] += "="

    with pytest.raises(ValueError, match="base64"):
        verify_uhp_response_bytes(raw, envelope, public_key=key.public_key())


def test_signature_rejects_envelope_tampering():
    key = Ed25519PrivateKey.generate()
    raw = response_bytes()
    envelope = sign_uhp_response_bytes(raw, private_key=key)
    envelope["preimage_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="preimage"):
        verify_uhp_response_bytes(raw, envelope, public_key=key.public_key())


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
    if os.name != "nt":
        private_path.chmod(0o600)
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    loaded_private = load_private_key(private_path)
    loaded_public = load_public_key(public_path)
    raw = response_bytes()
    envelope = sign_uhp_response_bytes(raw, private_key=loaded_private)

    verified = verify_uhp_response_bytes(raw, envelope, public_key=loaded_public)
    assert verified["key_id"] == public_key_id(private.public_key())


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_private_key_loader_rejects_group_readable_file(tmp_path):
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "producer-private.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o640)

    with pytest.raises(ValueError, match="group/other"):
        load_private_key(private_path)


def test_key_loaders_reject_symlinks(tmp_path):
    private = Ed25519PrivateKey.generate()
    private_target = tmp_path / "private-target.pem"
    private_target.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    if os.name != "nt":
        private_target.chmod(0o600)
    private_link = tmp_path / "private-link.pem"
    private_link.symlink_to(private_target)

    public_target = tmp_path / "public-target.pem"
    public_target.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    public_link = tmp_path / "public-link.pem"
    public_link.symlink_to(public_target)

    with pytest.raises(ValueError, match="non-symlink"):
        load_private_key(private_link)
    with pytest.raises(ValueError, match="non-symlink"):
        load_public_key(public_link)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_public_key_loader_rejects_group_writable_file(tmp_path):
    private = Ed25519PrivateKey.generate()
    public_path = tmp_path / "producer-public.pem"
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    public_path.chmod(0o664)

    with pytest.raises(ValueError, match="writable"):
        load_public_key(public_path)
