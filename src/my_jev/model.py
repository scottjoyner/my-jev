from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .schema import DecisionRecord, QuestionSpec, QuestionType


@dataclass(frozen=True)
class QuestionOutput:
    record_index: int
    name: str
    type: QuestionType
    options: list[str]
    logits: Tensor

    @property
    def probabilities(self) -> Tensor:
        return torch.softmax(self.logits, dim=-1)

    def probabilities_at_temperature(self, temperature: float) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return torch.softmax(self.logits / temperature, dim=-1)


class DynamicDecisionHead(nn.Module):
    """Legacy candidate-by-candidate cross-attention head."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            divisors = [
                n
                for n in range(min(num_heads, hidden_size), 0, -1)
                if hidden_size % n == 0
            ]
            num_heads = divisors[0]

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        state_hidden: Tensor,
        state_mask: Tensor,
        candidate_pooled: Tensor,
        candidate_state_index: Tensor,
    ) -> Tensor:
        selected_state = state_hidden.index_select(0, candidate_state_index)
        selected_mask = state_mask.index_select(0, candidate_state_index)
        query = candidate_pooled.unsqueeze(1)

        attended, _ = self.cross_attention(
            query=query,
            key=selected_state,
            value=selected_state,
            key_padding_mask=~selected_mask.bool(),
            need_weights=False,
        )
        context = self.norm(attended.squeeze(1) + candidate_pooled)
        features = torch.cat(
            [
                candidate_pooled,
                context,
                candidate_pooled * context,
                torch.abs(candidate_pooled - context),
            ],
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)


class OptionQueryDecisionHead(nn.Module):
    """Score runtime options as learned queries over shared state token memory.

    The design is inspired by the independent MIT-licensed
    vinnylarouge/jevlike option-attention pattern, generalized here to multiple
    typed questions per state.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        rank: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rank = int(rank or min(hidden_size, 256))
        if self.rank < 1:
            raise ValueError("rank must be >= 1")

        self.context_norm = nn.LayerNorm(hidden_size)
        self.option_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, self.rank, bias=False)
        self.key = nn.Linear(hidden_size, self.rank, bias=False)
        self.value = nn.Linear(hidden_size, self.rank, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state_hidden: Tensor,
        state_mask: Tensor,
        candidate_padded: Tensor,
        candidate_mask: Tensor,
        *,
        shuffle_state: bool = False,
    ) -> Tensor:
        if state_hidden.ndim != 3 or candidate_padded.ndim != 3:
            raise ValueError("state and candidate tensors must be rank 3")
        if state_hidden.shape[0] != candidate_padded.shape[0]:
            raise ValueError("state and candidate batch sizes must match")

        context = self.context_norm(state_hidden)
        candidates = self.option_norm(candidate_padded)

        if shuffle_state and context.shape[0] > 1:
            context = context.roll(1, dims=0)
            state_mask = state_mask.roll(1, dims=0)

        query = self.query(candidates)
        key = self.key(context)
        value = self.value(context)

        attention_logits = torch.einsum(
            "bcr,blr->bcl",
            query,
            key,
        ) / math.sqrt(self.rank)
        attention_logits = attention_logits.masked_fill(
            ~state_mask[:, None, :].bool(),
            torch.finfo(attention_logits.dtype).min,
        )
        attention = torch.softmax(
            attention_logits.float(),
            dim=-1,
        ).to(value.dtype)
        attention = self.dropout(attention)

        attended = torch.einsum(
            "bcl,blr->bcr",
            attention,
            value,
        )
        logits = (
            query * attended
        ).sum(-1) / math.sqrt(self.rank)
        return logits.masked_fill(
            ~candidate_mask.bool(),
            torch.finfo(logits.dtype).min,
        )


class SystemOneModel(nn.Module):
    def __init__(
        self,
        backbone: str = "answerdotai/ModernBERT-base",
        *,
        max_state_length: int = 2048,
        max_candidate_length: int = 192,
        dropout: float = 0.1,
        num_heads: int = 8,
        head_kind: str = "option_query",
        head_rank: int | None = None,
    ):
        super().__init__()
        if head_kind not in {"option_query", "legacy"}:
            raise ValueError("head_kind must be option_query or legacy")

        self.backbone_name = backbone
        self.max_state_length = max_state_length
        self.max_candidate_length = max_candidate_length
        self.head_kind = head_kind
        self.head_rank = head_rank
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(backbone)
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(backbone)
        hidden_size = int(self.encoder.config.hidden_size)

        if head_kind == "option_query":
            self.head = OptionQueryDecisionHead(
                hidden_size,
                rank=head_rank,
                dropout=dropout,
            )
            self.head_rank = self.head.rank
        else:
            self.head = DynamicDecisionHead(
                hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            )

    @staticmethod
    def _mean_pool(hidden: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * weights).sum(dim=1)
        return summed / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _candidate_text(
        question: QuestionSpec,
        option: str,
        option_index: int,
    ) -> str:
        if question.type == QuestionType.NOUL:
            return f"Question: {question.instructions}\nAnswer: {option}"
        if question.type == QuestionType.SCORE:
            total = len(question.options or [])
            return (
                f"Question: {question.instructions}\n"
                f"Ordered level {option_index + 1} of {total}: {option}"
            )
        return f"Question: {question.instructions}\nOption: {option}"

    def forward_records(
        self,
        records: Iterable[DecisionRecord],
        *,
        shuffle_state: bool = False,
    ) -> list[QuestionOutput]:
        records = list(records)
        if not records:
            return []

        state_texts = [record.state for record in records]
        candidate_texts: list[str] = []
        candidate_state_indices: list[int] = []
        per_record_counts = [0 for _ in records]
        groups: list[
            tuple[int, str, QuestionSpec, int, int, int, int]
        ] = []

        for record_index, record in enumerate(records):
            for name, question in record.questions.items():
                global_start = len(candidate_texts)
                local_start = per_record_counts[record_index]
                for option_index, option in enumerate(
                    question.options or []
                ):
                    candidate_texts.append(
                        self._candidate_text(
                            question,
                            option,
                            option_index,
                        )
                    )
                    candidate_state_indices.append(record_index)
                    per_record_counts[record_index] += 1
                global_end = len(candidate_texts)
                local_end = per_record_counts[record_index]
                groups.append(
                    (
                        record_index,
                        name,
                        question,
                        local_start,
                        local_end,
                        global_start,
                        global_end,
                    )
                )

        device = next(self.parameters()).device
        states = self.tokenizer(
            state_texts,
            padding=True,
            truncation=True,
            max_length=self.max_state_length,
            return_tensors="pt",
        ).to(device)
        candidates = self.tokenizer(
            candidate_texts,
            padding=True,
            truncation=True,
            max_length=self.max_candidate_length,
            return_tensors="pt",
        ).to(device)

        state_result = self.encoder(**states)
        candidate_result = self.encoder(**candidates)
        candidate_pooled = self._mean_pool(
            candidate_result.last_hidden_state,
            candidates["attention_mask"],
        )

        if self.head_kind == "option_query":
            rows: list[Tensor] = []
            offset = 0
            for count in per_record_counts:
                rows.append(
                    candidate_pooled[offset : offset + count]
                )
                offset += count
            candidate_padded = pad_sequence(
                rows,
                batch_first=True,
            )
            width = candidate_padded.shape[1]
            candidate_mask = (
                torch.arange(
                    width,
                    device=device,
                )[None, :]
                < torch.tensor(
                    per_record_counts,
                    device=device,
                )[:, None]
            )
            all_logits = self.head(
                state_hidden=state_result.last_hidden_state,
                state_mask=states["attention_mask"],
                candidate_padded=candidate_padded,
                candidate_mask=candidate_mask,
                shuffle_state=shuffle_state,
            )
        else:
            state_hidden = state_result.last_hidden_state
            state_mask = states["attention_mask"]
            if shuffle_state and state_hidden.shape[0] > 1:
                state_hidden = state_hidden.roll(1, dims=0)
                state_mask = state_mask.roll(1, dims=0)
            candidate_state_index = torch.tensor(
                candidate_state_indices,
                device=device,
                dtype=torch.long,
            )
            all_logits = self.head(
                state_hidden=state_hidden,
                state_mask=state_mask,
                candidate_pooled=candidate_pooled,
                candidate_state_index=candidate_state_index,
            )

        outputs: list[QuestionOutput] = []
        for (
            record_index,
            name,
            question,
            local_start,
            local_end,
            global_start,
            global_end,
        ) in groups:
            if self.head_kind == "option_query":
                logits = all_logits[
                    record_index,
                    local_start:local_end,
                ]
            else:
                logits = all_logits[
                    global_start:global_end
                ]
            outputs.append(
                QuestionOutput(
                    record_index=record_index,
                    name=name,
                    type=question.type,
                    options=list(question.options or []),
                    logits=logits,
                )
            )
        return outputs

    def predict(
        self,
        records: Iterable[DecisionRecord],
        *,
        temperature: float = 1.0,
    ) -> list[dict[str, object]]:
        self.eval()
        with torch.inference_mode():
            outputs = self.forward_records(records)

        result: list[dict[str, object]] = []
        for output in outputs:
            probabilities = (
                output.probabilities_at_temperature(temperature)
                .detach()
                .cpu()
            )
            best = int(probabilities.argmax().item())
            payload: dict[str, object] = {
                "record_index": output.record_index,
                "question": output.name,
                "type": output.type.value,
                "options": output.options,
                "probabilities": probabilities.tolist(),
                "choice": output.options[best],
                "confidence": float(
                    probabilities[best].item()
                ),
                "temperature": temperature,
            }
            if output.type == QuestionType.NOUL:
                payload["noul"] = float(
                    probabilities[1].item()
                )
            elif output.type == QuestionType.SCORE:
                positions = torch.linspace(
                    0.0,
                    1.0,
                    len(output.options),
                )
                payload["score"] = float(
                    (positions * probabilities).sum().item()
                )
            result.append(payload)
        return result
