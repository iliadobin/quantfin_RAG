"""
LLM-direct baselines (no retrieval, no citations).

These baselines use the same DeepSeek API client as the RAG generators, but do
not consult the local corpus. They are useful as "how good is the LLM alone?"
reference points in the benchmark matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from knowledge.models import Answer

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a quantitative finance QA assistant.\n"
    "You do NOT have access to any local PDF corpus or retrieval results.\n"
    "Answer from your general knowledge only.\n"
    "If you are unsure, say so and ask a clarifying question or refuse.\n"
    "Do NOT fabricate citations or claim you read specific documents/pages.\n"
)


def _detect_refusal(text: str) -> Optional[str]:
    """Lightweight refusal detection to map free-form LLM output to Answer fields."""
    t = (text or "").lower()
    patterns = [
        "i cannot answer",
        "i can't answer",
        "i do not have enough information",
        "i don't have enough information",
        "insufficient information",
        "cannot be answered",
        "outside the scope",
        "not enough context",
        "unclear",
        "ambiguous",
    ]
    return "llm_refusal" if any(p in t for p in patterns) else None


@dataclass
class LLMDirectBaseline:
    """
    Direct LLM baseline implementing the same `.run()` contract as RAG pipelines.

    Notes:
    - Returns `Answer` with empty citations (by design).
    - Initializes DeepSeek client lazily to avoid import-time failure when
      `DEEPSEEK_API_KEY` is not set.
    """

    model: str = "deepseek-chat"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.0
    max_tokens: int = 1200
    cache_enabled: bool = True
    name_override: Optional[str] = None
    version_override: str = "1.0"

    _llm_client: Any = None  # DeepSeekClient, lazily created

    def _get_client(self):
        if self._llm_client is not None:
            return self._llm_client

        # Local import so that just importing this module does not require env setup.
        from llm.deepseek_client import DeepSeekClient

        try:
            self._llm_client = DeepSeekClient(cache_enabled=self.cache_enabled)
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize DeepSeekClient for LLM-direct baseline. "
                "Make sure `DEEPSEEK_API_KEY` is set (see env.example)."
            ) from e
        return self._llm_client

    def run(
        self,
        query: str,
        corpus_profile: str = "public",
        **kwargs,
    ) -> Answer:
        """
        Run LLM-direct baseline.

        Args:
            query: user question
            corpus_profile: ignored (present for interface compatibility)
            **kwargs: supports `temperature`, `max_tokens`, `use_cache`
        """
        client = self._get_client()

        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        use_cache = bool(kwargs.get("use_cache", True))

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            resp: Dict[str, Any] = client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                use_cache=use_cache,
            )
            text = (resp.get("content") or "").strip()
        except Exception as e:
            logger.exception("LLM-direct generation failed")
            return Answer(
                text="Error generating answer (LLM-direct baseline).",
                citations=[],
                confidence=0.0,
                refusal_reason="generation_error",
                metadata={"error": str(e), "model": self.model},
            )

        refusal_reason = _detect_refusal(text)
        confidence = 0.0 if refusal_reason else 0.55

        return Answer(
            text=text,
            citations=[],
            confidence=confidence,
            refusal_reason=refusal_reason,
            metadata={
                "baseline": "llm_direct",
                "model": resp.get("model") or self.model,
                "usage": resp.get("usage"),
                "corpus_profile_ignored": corpus_profile,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    @property
    def name(self) -> str:
        return self.name_override or f"Baseline_LLM_Direct_{self.model}"

    @property
    def version(self) -> str:
        return self.version_override


class DeepSeekChatDirectBaseline(LLMDirectBaseline):
    def __init__(self, **kwargs):
        super().__init__(
            model="deepseek-chat",
            name_override="Baseline_LLM_Direct_Chat",
            **kwargs,
        )


class DeepSeekReasonerDirectBaseline(LLMDirectBaseline):
    def __init__(self, **kwargs):
        super().__init__(
            model="deepseek-reasoner",
            name_override="Baseline_LLM_Direct_Reasoner",
            **kwargs,
        )


