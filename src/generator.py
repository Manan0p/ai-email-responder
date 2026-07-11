import re
import time
from dataclasses import dataclass
from typing import List, Optional
from groq import Groq
from src.config import config
from src.retriever import Retriever, RetrievedContext

MAX_RETRIES = 5
BASE_DELAY = 5


def _retry_delay_from_error(exc: Exception) -> float:
    msg = str(exc)
    match = re.search(r"retry in (\d+(?:\.\d+)?)", msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1
    return 0


SYSTEM_PROMPT = """You are a professional email assistant. Based on similar past emails and their replies, generate an appropriate response to the new incoming email.

Guidelines:
- Match the professional tone of the past examples
- Address all points raised in the email
- Be concise but thorough
- Include appropriate greeting and sign-off
- Do not include a subject line — only generate the reply body"""


def _build_prompt(context: RetrievedContext, from_addr: str, subject: str, body: str) -> str:
    return (
        f"## Past Email Examples (for reference):\n\n"
        f"{context.format_for_prompt()}\n\n"
        f"---\n\n"
        f"## New Incoming Email:\n"
        f"From: {from_addr}\n"
        f"Subject: {subject}\n"
        f"Body:\n{body}\n\n"
        f"---\n\n"
        f"Generate the reply:"
    )


@dataclass
class GeneratedReply:
    reply_text: str
    retrieved_ids: List[str]
    model: str
    latency_seconds: float
    prompt: str = ""


class Generator:
    def __init__(self, retriever: Retriever = None):
        config.validate()
        self.retriever = retriever or Retriever()
        self.client = Groq(api_key=config.groq_api_key)

    def generate(
        self,
        from_addr: str,
        subject: str,
        body: str,
        exclude_ids: Optional[List[str]] = None,
    ) -> GeneratedReply:
        email_text = f"Subject: {subject}\n\n{body}"
        context = self.retriever.retrieve(email_text, exclude_ids=exclude_ids)

        prompt = _build_prompt(context, from_addr, subject, body)

        start = time.time()
        last_exc = None
        for attempt in range(MAX_RETRIES):
            time.sleep(2.1)  # Buffer for Groq 30 RPM limit
            try:
                response = self.client.chat.completions.create(
                    model=config.generation_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.temperature,
                )
                latency = time.time() - start
                return GeneratedReply(
                    reply_text=response.choices[0].message.content.strip(),
                    retrieved_ids=[p.id for p in context.email_pairs],
                    model=config.generation_model,
                    latency_seconds=round(latency, 3),
                    prompt=prompt,
                )
            except Exception as exc:
                last_exc = exc
                if "429" in str(exc) or "rate limit" in str(exc).lower():
                    delay = _retry_delay_from_error(exc) or BASE_DELAY * (2 ** attempt)
                    print(f"  [Rate limited] Retrying in {delay:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(delay)
                else:
                    raise
        raise last_exc  # type: ignore[misc]
