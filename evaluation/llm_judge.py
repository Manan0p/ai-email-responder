import json
import re
import time
from typing import Dict, Any
from dataclasses import dataclass
from groq import Groq
from src.config import config

MAX_RETRIES = 5
BASE_DELAY = 5


def _retry_delay_from_error(exc: Exception) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1
    return 0


JUDGE_PROMPT = """You are an expert email quality evaluator. You will evaluate an AI-generated email reply by comparing it to a human-written reference reply.

## Incoming Email:
From: {from_addr}
Subject: {subject}
Body:
{body}

## Reference (Human-Written) Reply:
{reference_reply}

## AI-Generated Reply:
{generated_reply}

## Evaluation Instructions:
Score the AI-generated reply on these 5 dimensions using a 1-5 scale:

1. **SEMANTIC_ACCURACY** (1-5): Does the AI reply convey the same core message and meaning as the reference?
   - 1: Completely different message
   - 3: Partially captures the message
   - 5: Captures the full meaning, even if worded differently

2. **INTENT_ALIGNMENT** (1-5): Does the AI reply address the same goals and actions as the reference?
   - 1: Addresses completely wrong intent
   - 3: Addresses some goals but misses key ones
   - 5: Addresses all the same goals/actions

3. **TONE** (1-5): Is the tone appropriate for this email context?
   - 1: Completely inappropriate tone
   - 3: Acceptable but not ideal tone
   - 5: Perfect tone for the context

4. **COMPLETENESS** (1-5): Does the AI reply address all points raised in the incoming email?
   - 1: Misses most points
   - 3: Addresses about half the points
   - 5: Addresses every point

5. **COHERENCE** (1-5): Is the reply well-structured, grammatically correct, and professional?
   - 1: Incoherent or very poorly written
   - 3: Readable but has issues
   - 5: Excellent writing quality

Respond with ONLY a JSON object in this exact format:
{{
  "semantic_accuracy": {{"score": <1-5>, "reason": "<brief justification>"}},
  "intent_alignment": {{"score": <1-5>, "reason": "<brief justification>"}},
  "tone": {{"score": <1-5>, "reason": "<brief justification>"}},
  "completeness": {{"score": <1-5>, "reason": "<brief justification>"}},
  "coherence": {{"score": <1-5>, "reason": "<brief justification>"}}
}}"""


@dataclass
class JudgeScore:
    semantic_accuracy: Dict[str, Any]
    intent_alignment: Dict[str, Any]
    tone: Dict[str, Any]
    completeness: Dict[str, Any]
    coherence: Dict[str, Any]

    @property
    def average(self) -> float:
        scores = [
            self.semantic_accuracy["score"],
            self.intent_alignment["score"],
            self.tone["score"],
            self.completeness["score"],
            self.coherence["score"],
        ]
        return sum(scores) / len(scores)

    @property
    def normalized_average(self) -> float:
        return (self.average - 1.0) / 4.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_accuracy": self.semantic_accuracy,
            "intent_alignment": self.intent_alignment,
            "tone": self.tone,
            "completeness": self.completeness,
            "coherence": self.coherence,
            "average_score": round(self.average, 2),
            "normalized_average": round(self.normalized_average, 4),
        }


class LLMJudge:
    def __init__(self):
        config.validate()
        self.client = Groq(api_key=config.groq_api_key)

    def evaluate(
        self,
        incoming_from: str,
        incoming_subject: str,
        incoming_body: str,
        reference_reply: str,
        generated_reply: str,
    ) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(
            from_addr=incoming_from,
            subject=incoming_subject,
            body=incoming_body,
            reference_reply=reference_reply,
            generated_reply=generated_reply,
        )

        last_exc = None
        for attempt in range(MAX_RETRIES):
            time.sleep(2.1)  # Groq rate limit buffer
            try:
                response = self.client.chat.completions.create(
                    model=config.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )

                text = response.choices[0].message.content.strip()
                data = json.loads(text)

                return JudgeScore(
                    semantic_accuracy=data["semantic_accuracy"],
                    intent_alignment=data["intent_alignment"],
                    tone=data["tone"],
                    completeness=data["completeness"],
                    coherence=data["coherence"],
                )
            except Exception as exc:
                last_exc = exc
                if "429" in str(exc) or "rate limit" in str(exc).lower():
                    delay = _retry_delay_from_error(exc) or BASE_DELAY * (2 ** attempt)
                    print(f"  [Rate limited] Judge retrying in {delay:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(delay)
                else:
                    raise
        raise last_exc  # type: ignore[misc]
