from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from .model import QuestionOutput
from .schema import (
    DecisionRecord,
    QuestionSpec,
    QuestionType,
)


def causal_candidate_text(
    state: str,
    question: QuestionSpec,
    option: str,
    option_index: int,
) -> str:
    if question.type == QuestionType.SCORE:
        total = len(
            question.options or []
        )
        answer = (
            f"ordered level "
            f"{option_index + 1} "
            f"of {total}: {option}"
        )
    elif question.type == QuestionType.NOUL:
        answer = (
            f"binary answer: {option}"
        )
    else:
        answer = (
            f"choice option: {option}"
        )

    return (
        "You are a bounded decision scorer. "
        "Score how well one candidate answers "
        "the typed question for the supplied state.\n\n"
        f"STATE:\n{state}\n\n"
        f"QUESTION TYPE:\n{question.type.value}\n\n"
        f"QUESTION:\n{question.instructions}\n\n"
        f"CANDIDATE:\n{answer}\n"
    )


class CausalScalarSystemOneModel(nn.Module):
    """Independent scalar scorer over (state, question, option) sequences.

    This is the correctness-first causal lane.  It intentionally re-encodes the
    full state for each candidate.  Shared-state cache branching is a later
    inference optimization and must reproduce this scorer's logits within a
    tested tolerance before replacing it.
    """

    head_kind = "causal_scalar"
    head_rank = None

    def __init__(
        self,
        backbone: str = (
            "Qwen/Qwen3.5-4B-Base"
        ),
        *,
        max_length: int = 1024,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: str = "all-linear",
        enable_lora: bool = True,
        adapter_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.max_length = max_length
        self.max_state_length = max_length
        self.max_candidate_length = max_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules

        self.tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(
                backbone
            )
        )
        if (
            self.tokenizer.pad_token_id
            is None
        ):
            if (
                self.tokenizer.eos_token_id
                is None
            ):
                raise ValueError(
                    "causal scorer tokenizer "
                    "needs a pad or EOS token"
                )
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        load_kwargs: dict[
            str,
            object,
        ] = {
            "num_labels": 1,
            "problem_type": "regression",
        }
        if dtype is not None:
            load_kwargs[
                "torch_dtype"
            ] = dtype

        base = (
            AutoModelForSequenceClassification
            .from_pretrained(
                backbone,
                **load_kwargs,
            )
        )
        base.config.pad_token_id = (
            self.tokenizer.pad_token_id
        )

        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "loading a causal scalar "
                    "LoRA checkpoint requires "
                    "the 'causal' extra"
                ) from exc
            self.encoder = (
                PeftModel.from_pretrained(
                    base,
                    str(adapter_path),
                    is_trainable=False,
                )
            )
        elif (
            enable_lora
            and lora_r > 0
        ):
            try:
                from peft import (
                    LoraConfig,
                    TaskType,
                    get_peft_model,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "training the causal scalar "
                    "lane requires the "
                    "'causal' extra"
                ) from exc

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=(
                    lora_dropout
                ),
                target_modules=(
                    target_modules
                ),
                modules_to_save=[
                    "score"
                ],
                bias="none",
                task_type=(
                    TaskType.SEQ_CLS
                ),
            )
            self.encoder = (
                get_peft_model(
                    base,
                    config,
                )
            )
        else:
            self.encoder = base

        self.to(device)

    def gradient_checkpointing_enable(
        self,
    ) -> None:
        enable = getattr(
            self.encoder,
            "gradient_checkpointing_enable",
            None,
        )
        if not callable(enable):
            raise RuntimeError(
                "selected causal backbone "
                "does not expose gradient "
                "checkpointing"
            )
        enable()
        config = getattr(
            self.encoder,
            "config",
            None,
        )
        if config is not None:
            config.use_cache = False

    def _candidate_rows(
        self,
        records: list[DecisionRecord],
        *,
        shuffle_state: bool,
    ) -> tuple[
        list[str],
        list[
            tuple[
                int,
                str,
                QuestionSpec,
                int,
                int,
            ]
        ],
    ]:
        texts: list[str] = []
        groups: list[
            tuple[
                int,
                str,
                QuestionSpec,
                int,
                int,
            ]
        ] = []
        state_count = len(records)

        for record_index, record in enumerate(
            records
        ):
            state_index = record_index
            if (
                shuffle_state
                and state_count > 1
            ):
                state_index = (
                    record_index - 1
                ) % state_count
            state = records[
                state_index
            ].state

            for name, question in (
                record.questions.items()
            ):
                start = len(texts)
                for (
                    option_index,
                    option,
                ) in enumerate(
                    question.options or []
                ):
                    texts.append(
                        causal_candidate_text(
                            state,
                            question,
                            option,
                            option_index,
                        )
                    )
                groups.append(
                    (
                        record_index,
                        name,
                        question,
                        start,
                        len(texts),
                    )
                )
        return texts, groups

    def forward_records(
        self,
        records: Iterable[
            DecisionRecord
        ],
        *,
        shuffle_state: bool = False,
    ) -> list[QuestionOutput]:
        records = list(records)
        if not records:
            return []

        texts, groups = (
            self._candidate_rows(
                records,
                shuffle_state=(
                    shuffle_state
                ),
            )
        )
        device = next(
            self.parameters()
        ).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        result = self.encoder(
            **encoded
        )
        logits = (
            result.logits
            .reshape(-1)
        )

        outputs: list[
            QuestionOutput
        ] = []
        for (
            record_index,
            name,
            question,
            start,
            end,
        ) in groups:
            outputs.append(
                QuestionOutput(
                    record_index=(
                        record_index
                    ),
                    name=name,
                    type=question.type,
                    options=list(
                        question.options
                        or []
                    ),
                    logits=logits[
                        start:end
                    ],
                )
            )
        return outputs

    def predict(
        self,
        records: Iterable[
            DecisionRecord
        ],
        *,
        temperature: float = 1.0,
    ) -> list[
        dict[str, object]
    ]:
        self.eval()
        with torch.inference_mode():
            outputs = (
                self.forward_records(
                    records
                )
            )

        result: list[
            dict[str, object]
        ] = []
        for output in outputs:
            probabilities = (
                output
                .probabilities_at_temperature(
                    temperature
                )
                .detach()
                .cpu()
            )
            best = int(
                probabilities
                .argmax()
                .item()
            )
            payload: dict[
                str,
                object,
            ] = {
                "record_index": (
                    output.record_index
                ),
                "question": output.name,
                "type": (
                    output.type.value
                ),
                "options": (
                    output.options
                ),
                "probabilities": (
                    probabilities
                    .tolist()
                ),
                "choice": (
                    output.options[
                        best
                    ]
                ),
                "confidence": float(
                    probabilities[
                        best
                    ].item()
                ),
                "temperature": (
                    temperature
                ),
            }
            if (
                output.type
                == QuestionType.NOUL
            ):
                payload["noul"] = float(
                    probabilities[
                        1
                    ].item()
                )
            elif (
                output.type
                == QuestionType.SCORE
            ):
                positions = (
                    torch.linspace(
                        0.0,
                        1.0,
                        len(
                            output.options
                        ),
                    )
                )
                payload["score"] = float(
                    (
                        positions
                        * probabilities
                    )
                    .sum()
                    .item()
                )
            result.append(
                payload
            )
        return result

    def save_checkpoint(
        self,
        output_dir: str | Path,
        *,
        extra: dict | None = None,
    ) -> None:
        output = Path(
            output_dir
        )
        output.mkdir(
            parents=True,
            exist_ok=True,
        )
        model_dir = (
            output / "model"
        )
        self.encoder.save_pretrained(
            model_dir
        )
        self.tokenizer.save_pretrained(
            output / "tokenizer"
        )
        config = {
            "model_type": (
                "causal_scalar"
            ),
            "backbone": (
                self.backbone_name
            ),
            "max_length": (
                self.max_length
            ),
            "head_kind": (
                self.head_kind
            ),
            "head_rank": None,
            "lora_r": self.lora_r,
            "lora_alpha": (
                self.lora_alpha
            ),
            "lora_dropout": (
                self.lora_dropout
            ),
            "target_modules": (
                self.target_modules
            ),
            "extra": extra or {},
        }
        (
            output
            / "my_jev_config.json"
        ).write_text(
            json.dumps(
                config,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def load_causal_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> CausalScalarSystemOneModel:
    root = Path(path)
    config = json.loads(
        (
            root
            / "my_jev_config.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    selected_device = (
        torch.device(device)
    )
    dtype = (
        torch.bfloat16
        if (
            selected_device.type
            == "cuda"
            and torch.cuda
            .is_bf16_supported()
        )
        else None
    )
    return CausalScalarSystemOneModel(
        backbone=config[
            "backbone"
        ],
        max_length=int(
            config[
                "max_length"
            ]
        ),
        lora_r=int(
            config.get(
                "lora_r",
                16,
            )
        ),
        lora_alpha=int(
            config.get(
                "lora_alpha",
                32,
            )
        ),
        lora_dropout=float(
            config.get(
                "lora_dropout",
                0.05,
            )
        ),
        target_modules=str(
            config.get(
                "target_modules",
                "all-linear",
            )
        ),
        enable_lora=False,
        adapter_path=(
            root / "model"
        ),
        device=selected_device,
        dtype=dtype,
    )
