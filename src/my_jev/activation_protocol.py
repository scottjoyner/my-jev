from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
import torch
from torch import Tensor


FRAME_VERSION = (
    "my-jev-activation-frame-v1"
)
_HEADER_PREFIX = struct.Struct(
    ">I"
)
_SUPPORTED_DTYPES = {
    "float16": np.dtype(
        np.float16
    ),
    "float32": np.dtype(
        np.float32
    ),
}


@dataclass(frozen=True)
class ActivationFrame:
    provider: str
    model_sha256: str
    runtime_revision: str
    prompt_contract: str
    state_taps: Tensor
    state_mask: Tensor
    option_taps: Tensor
    option_mask: Tensor

    @property
    def hidden_size(
        self,
    ) -> int:
        return int(
            self.state_taps.shape[
                -1
            ]
        )

    @property
    def tap_count(
        self,
    ) -> int:
        return int(
            self.state_taps.shape[
                1
            ]
        )

    def validate(
        self,
    ) -> None:
        if (
            self.state_taps.ndim
            != 4
        ):
            raise ValueError(
                "state_taps must be "
                "[batch,taps,tokens,hidden]"
            )
        if (
            self.option_taps.ndim
            != 4
        ):
            raise ValueError(
                "option_taps must be "
                "[batch,options,taps,hidden]"
            )
        if (
            self.state_mask.ndim
            != 2
        ):
            raise ValueError(
                "state_mask must be "
                "[batch,tokens]"
            )
        if (
            self.option_mask.ndim
            != 2
        ):
            raise ValueError(
                "option_mask must be "
                "[batch,options]"
            )

        batch = int(
            self.state_taps.shape[
                0
            ]
        )
        taps = int(
            self.state_taps.shape[
                1
            ]
        )
        tokens = int(
            self.state_taps.shape[
                2
            ]
        )
        hidden = int(
            self.state_taps.shape[
                3
            ]
        )

        if (
            self.option_taps.shape[
                0
            ]
            != batch
        ):
            raise ValueError(
                "option batch does not "
                "match state batch"
            )
        if (
            self.option_taps.shape[
                2
            ]
            != taps
        ):
            raise ValueError(
                "option tap count does "
                "not match state tap count"
            )
        if (
            self.option_taps.shape[
                3
            ]
            != hidden
        ):
            raise ValueError(
                "option hidden size does "
                "not match state hidden size"
            )
        if tuple(
            self.state_mask.shape
        ) != (
            batch,
            tokens,
        ):
            raise ValueError(
                "state mask shape does not "
                "match state activations"
            )
        if tuple(
            self.option_mask.shape
        ) != (
            batch,
            int(
                self.option_taps.shape[
                    1
                ]
            ),
        ):
            raise ValueError(
                "option mask shape does not "
                "match option activations"
            )

        for name, value in (
            (
                "provider",
                self.provider,
            ),
            (
                "model_sha256",
                self.model_sha256,
            ),
            (
                "runtime_revision",
                self.runtime_revision,
            ),
            (
                "prompt_contract",
                self.prompt_contract,
            ),
        ):
            if not value:
                raise ValueError(
                    f"{name} must be non-empty"
                )


def _tensor_numpy(
    tensor: Tensor,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    return np.asarray(
        tensor.detach()
        .cpu()
        .numpy(),
        dtype=dtype,
        order="C",
    )


def write_activation_frame(
    stream: BinaryIO,
    frame: ActivationFrame,
    *,
    dtype: str = "float16",
) -> None:
    frame.validate()

    if dtype not in (
        _SUPPORTED_DTYPES
    ):
        raise ValueError(
            f"unsupported activation dtype: {dtype}"
        )
    np_dtype = (
        _SUPPORTED_DTYPES[
            dtype
        ]
    )

    state = _tensor_numpy(
        frame.state_taps,
        dtype=np_dtype,
    )
    options = _tensor_numpy(
        frame.option_taps,
        dtype=np_dtype,
    )
    state_mask = _tensor_numpy(
        frame.state_mask,
        dtype=np.dtype(
            np.uint8
        ),
    )
    option_mask = _tensor_numpy(
        frame.option_mask,
        dtype=np.dtype(
            np.uint8
        ),
    )

    header = {
        "version": (
            FRAME_VERSION
        ),
        "provider": (
            frame.provider
        ),
        "model_sha256": (
            frame.model_sha256
        ),
        "runtime_revision": (
            frame.runtime_revision
        ),
        "prompt_contract": (
            frame.prompt_contract
        ),
        "dtype": dtype,
        "state_shape": list(
            state.shape
        ),
        "state_mask_shape": list(
            state_mask.shape
        ),
        "option_shape": list(
            options.shape
        ),
        "option_mask_shape": list(
            option_mask.shape
        ),
        "state_bytes": int(
            state.nbytes
        ),
        "state_mask_bytes": int(
            state_mask.nbytes
        ),
        "option_bytes": int(
            options.nbytes
        ),
        "option_mask_bytes": int(
            option_mask.nbytes
        ),
    }
    encoded = json.dumps(
        header,
        separators=(
            ",",
            ":",
        ),
        sort_keys=True,
    ).encode(
        "utf-8"
    )

    stream.write(
        _HEADER_PREFIX.pack(
            len(
                encoded
            )
        )
    )
    stream.write(
        encoded
    )
    stream.write(
        state.tobytes(
            order="C"
        )
    )
    stream.write(
        state_mask.tobytes(
            order="C"
        )
    )
    stream.write(
        options.tobytes(
            order="C"
        )
    )
    stream.write(
        option_mask.tobytes(
            order="C"
        )
    )


def _read_exact(
    stream: BinaryIO,
    size: int,
) -> bytes:
    value = stream.read(
        size
    )
    if (
        value is None
        or len(
            value
        )
        != size
    ):
        raise EOFError(
            "truncated activation frame"
        )
    return value


def read_activation_frame(
    stream: BinaryIO,
) -> ActivationFrame:
    prefix = _read_exact(
        stream,
        _HEADER_PREFIX.size,
    )
    (
        header_size,
    ) = _HEADER_PREFIX.unpack(
        prefix
    )
    if (
        header_size < 2
        or header_size
        > 1_000_000
    ):
        raise ValueError(
            "invalid activation frame "
            "header size"
        )

    header = json.loads(
        _read_exact(
            stream,
            header_size,
        ).decode(
            "utf-8"
        )
    )
    if (
        header.get(
            "version"
        )
        != FRAME_VERSION
    ):
        raise ValueError(
            "unsupported activation "
            "frame version"
        )

    dtype_name = str(
        header.get(
            "dtype"
        )
    )
    if dtype_name not in (
        _SUPPORTED_DTYPES
    ):
        raise ValueError(
            "unsupported activation "
            f"dtype: {dtype_name}"
        )
    dtype = (
        _SUPPORTED_DTYPES[
            dtype_name
        ]
    )

    state_shape = tuple(
        int(
            value
        )
        for value in (
            header[
                "state_shape"
            ]
        )
    )
    state_mask_shape = tuple(
        int(
            value
        )
        for value in (
            header[
                "state_mask_shape"
            ]
        )
    )
    option_shape = tuple(
        int(
            value
        )
        for value in (
            header[
                "option_shape"
            ]
        )
    )
    option_mask_shape = tuple(
        int(
            value
        )
        for value in (
            header[
                "option_mask_shape"
            ]
        )
    )

    state = np.frombuffer(
        _read_exact(
            stream,
            int(
                header[
                    "state_bytes"
                ]
            ),
        ),
        dtype=dtype,
    ).reshape(
        state_shape
    )
    state_mask = np.frombuffer(
        _read_exact(
            stream,
            int(
                header[
                    "state_mask_bytes"
                ]
            ),
        ),
        dtype=np.uint8,
    ).reshape(
        state_mask_shape
    )
    options = np.frombuffer(
        _read_exact(
            stream,
            int(
                header[
                    "option_bytes"
                ]
            ),
        ),
        dtype=dtype,
    ).reshape(
        option_shape
    )
    option_mask = np.frombuffer(
        _read_exact(
            stream,
            int(
                header[
                    "option_mask_bytes"
                ]
            ),
        ),
        dtype=np.uint8,
    ).reshape(
        option_mask_shape
    )

    frame = ActivationFrame(
        provider=str(
            header.get(
                "provider"
            )
            or ""
        ),
        model_sha256=str(
            header.get(
                "model_sha256"
            )
            or ""
        ),
        runtime_revision=str(
            header.get(
                "runtime_revision"
            )
            or ""
        ),
        prompt_contract=str(
            header.get(
                "prompt_contract"
            )
            or ""
        ),
        state_taps=(
            torch.from_numpy(
                state.copy()
            )
        ),
        state_mask=(
            torch.from_numpy(
                state_mask.copy()
            ).bool()
        ),
        option_taps=(
            torch.from_numpy(
                options.copy()
            )
        ),
        option_mask=(
            torch.from_numpy(
                option_mask.copy()
            ).bool()
        ),
    )
    frame.validate()
    return frame


def roundtrip_activation_frame(
    frame: ActivationFrame,
    *,
    dtype: str = "float16",
) -> ActivationFrame:
    buffer = io.BytesIO()
    write_activation_frame(
        buffer,
        frame,
        dtype=dtype,
    )
    buffer.seek(
        0
    )
    return read_activation_frame(
        buffer
    )
