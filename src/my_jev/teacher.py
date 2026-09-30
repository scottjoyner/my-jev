from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from .schema import DecisionRecord, TargetSpec


@dataclass(frozen=True)
class TeacherConfig:
    endpoint: str
    model: str
    temperature: float = 0.2
    timeout_seconds: int = 120
    api_key_env: str | None = None


def build_teacher_prompt(record: DecisionRecord) -> str:
    questions = {
        name: {
            "type": question.type.value,
            "instructions": question.instructions,
            "options": question.options,
        }
        for name, question in record.questions.items()
    }
    return (
        "You are labeling a bounded decision dataset. Return JSON only. "
        "For every question, return a probability distribution over the options in exactly "
        "the provided order. Probabilities must be non-negative and sum to 1. Do not add "
        "explanations.\n\n"
        f"STATE:\n{record.state}\n\n"
        f"QUESTIONS:\n{json.dumps(questions, ensure_ascii=False)}\n\n"
        'OUTPUT SHAPE: {"targets":{"question_name":{"distribution":[...]}}}'
    )


def parse_teacher_targets(
    text: str,
    record: DecisionRecord,
) -> dict[str, TargetSpec]:
    text = text.strip()
    if text.startswith(chr(96) * 3):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.startswith("json"):
            text = text[4:].lstrip()

    payload = json.loads(text)
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, dict):
        raise ValueError("teacher response must contain a targets object")

    targets: dict[str, TargetSpec] = {}
    for name, question in record.questions.items():
        if name not in raw_targets:
            raise ValueError(f"teacher omitted question: {name}")
        target = TargetSpec.model_validate(raw_targets[name])
        if target.distribution is None:
            raise ValueError(f"teacher must return a distribution for {name}")
        expected = len(question.options or [])
        if len(target.distribution) != expected:
            raise ValueError(
                f"teacher distribution length mismatch for {name}: "
                f"expected {expected}, got {len(target.distribution)}"
            )
        targets[name] = target

    extra = set(raw_targets) - set(record.questions)
    if extra:
        raise ValueError(f"teacher returned unknown questions: {sorted(extra)}")
    return targets


class OpenAICompatibleTeacher:
    def __init__(self, config: TeacherConfig):
        self.config = config

    def label(self, record: DecisionRecord) -> dict[str, TargetSpec]:
        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": build_teacher_prompt(record),
                }
            ],
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"

        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.config.timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))

        content = payload["choices"][0]["message"]["content"]
        return parse_teacher_targets(content, record)


def aggregate_teacher_targets(
    record: DecisionRecord,
    samples: list[dict[str, TargetSpec]],
) -> dict[str, TargetSpec]:
    if not samples:
        raise ValueError("at least one teacher sample is required")

    aggregated: dict[str, TargetSpec] = {}
    for name, question in record.questions.items():
        size = len(question.options or [])
        totals = [0.0] * size
        for sample in samples:
            distribution = sample[name].distribution
            if distribution is None or len(distribution) != size:
                raise ValueError(f"invalid teacher sample for {name}")
            for index, probability in enumerate(distribution):
                totals[index] += probability
        aggregated[name] = TargetSpec(
            distribution=[
                value / len(samples)
                for value in totals
            ]
        )
    return aggregated
