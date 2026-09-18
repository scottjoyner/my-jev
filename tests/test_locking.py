import pytest

from my_jev.locking import (
    Lease,
    ResourceBusy,
    ResourceRequest,
)


def test_exclusive_lease_blocks_second_owner(tmp_path):
    first = Lease(
        tmp_path,
        ResourceRequest("gpu"),
    )
    second = Lease(
        tmp_path,
        ResourceRequest("gpu"),
    )

    first.acquire()
    try:
        with pytest.raises(
            ResourceBusy
        ):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_lease_owner_record_is_removed_on_release(tmp_path):
    lease = Lease(
        tmp_path,
        ResourceRequest("gpu"),
    )
    lease.acquire()
    owner_path = lease.owner_path
    assert owner_path is not None
    assert owner_path.exists()

    lease.release()
    assert not owner_path.exists()
