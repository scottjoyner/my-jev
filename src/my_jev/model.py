from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
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


class DynamicDecisionHead(nn.Module):
    """Score runtime-defined options against one encoded state."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            divisors = [
                n for n in range(min(num_heads, hidden_size), 0, -1)
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


class SystemOneModel(nn.Module):
    def __init__(
        self,
        backbone: str = "answerdotai/ModernBERT-base",
        *,
        max_state_length: int = 2048,
        max_candidate_length: int = 192,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.max_state_length = max_state_length
        self.max_candidate_length = max_candidate_length
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(backbone)
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(backbone)
        hidden_size = int(self.encoder.config.hidden_size)
        self.head = DynamicDecisionHead(hidden_size, num_heads=num_heads, dropout=dropout)

    @staticmethod
    def _mean_pool(hidden: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * weights).sum(dim=1)
        return summed / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _candidate_text(question: QuestionSpec, option: str, option_index: int) -> str:
        if question.type == QuestionType.NOUL:
            return f"Question: {question.instructions}\nAnswer: {option}"
        if question.type == QuestionType.SCORE:
            total = len(question.options or [])
            return (
                f"Question: {question.instructions}\n"
                f"Ordered level {option_index + 1} of {total}: {option}"
            )
        return f"Question: {question.instructions}\nOption: {option}"

    def forward_records(self, records: Iterable[DecisionRecord]) -> list[QuestionOutput]:
        records = list(records)
        if not records:
            return []

        state_texts = [record.state for record in records]
        candidate_texts: list[str] = []
        candidate_state_indices: list[int] = []
        groups: list[tuple[int, str, QuestionSpec, int, int]] = []

        for record_index, record in enumerate(records):
            for name, question in record.questions.items():
                start = len(candidate_texts)
                for option_index, option in enumerate(question.options or []):
                    candidate_texts.append(self._candidate_text(question, option, option_index))
                    candidate_state_indices.append(record_index)
                end = len(candidate_texts)
                groups.append((record_index, name, question, start, end))

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
            candidate_result.last_hidden_state, candidates["attention_mask"]
        )
        candidate_state_index = torch.tensor(
            candidate_state_indices,
            device=device,
            dtype=torch.long,
        )

        all_logits = self.head(
            state_hidden=state_result.last_hidden_state,
            state_mask=states["attention_mask"],
            candidate_pooled=candidate_pooled,
            candidate_state_index=candidate_state_index,
        )

        outputs: list[QuestionOutput] = []
        for record_index, name, question, start, end in groups:
            outputs.append(
                QuestionOutput(
                    record_index=record_index,
                    name=name,
                    type=question.type,
                    options=list(question.options or []),
                    logits=all_logits[start:end],
                )
            )
        return outputs

    def predict(self, records: Iterable[DecisionRecord]) -> list[dict[str, object]]:
        self.eval()
        with torch.inference_mode():
            outputs = self.forward_records(records)

        result: list[dict[str, object]] = []
        for output in outputs:
            probabilities = output.probabilities.detach().cpu()
            best = int(probabilities.argmax().item())
            payload: dict[str, object] = {
                "record_index": output.record_index,
                "question": output.name,
                "type": output.type.value,
                "options": output.options,
                "probabilities": probabilities.tolist(),
                "choice": output.options[best],
                "confidence": float(probabilities[best].item()),
            }
            if output.type == QuestionType.NOUL:
                payload["noul"] = float(probabilities[1].item())
            elif output.type == QuestionType.SCORE:
                positions = torch.linspace(0.0, 1.0, len(output.options))
                payload["score"] = float((positions * probabilities).sum().item())
            result.append(payload)
        return result
