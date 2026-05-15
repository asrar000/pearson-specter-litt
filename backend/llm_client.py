"""
Shared Groq chat client.

The application uses Groq for all hosted LLM calls. This module keeps provider
details, API-key rotation, model fallback, and retry behavior in one place so
document parsing, drafting, and edit-pattern extraction stay focused on prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Optional

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

ROTATE_KEY_STATUSES = {401, 403, 429}
TRANSIENT_STATUSES = {408, 409, 425, 500, 502, 503, 504}
MODEL_FALLBACK_STATUSES = {400, 404}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str


class GroqAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class GroqLLMClient:
    def __init__(self) -> None:
        self._key_index = 0

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Call Groq chat completions with key rotation and model fallback.

        Rotation is attempted for auth failures, rate limits, transient HTTP
        failures, and transport errors. Model fallback is attempted when a model
        appears unavailable or incompatible.
        """
        keys = settings.GROQ_API_KEYS
        if not keys:
            raise RuntimeError(
                "Groq API key not configured. Set GROQ_API_KEY or GROQ_API_KEYS."
            )

        models = self._models(model)
        retries = max(1, settings.GROQ_RETRY_PER_MODEL)
        last_error: Optional[BaseException] = None

        for model_name in models:
            model_failed = False
            for attempt in range(retries):
                should_backoff = False

                for api_key in self._ordered_keys(keys):
                    try:
                        response = self._request(
                            api_key=api_key,
                            model=model_name,
                            system=system,
                            user=user,
                            max_tokens=max_tokens or settings.GROQ_MAX_TOKENS,
                            temperature=(
                                settings.GROQ_TEMPERATURE
                                if temperature is None
                                else temperature
                            ),
                        )
                        self._remember_successful_key(keys, api_key)
                        return response

                    except GroqAPIError as exc:
                        last_error = exc

                        if exc.status_code in MODEL_FALLBACK_STATUSES:
                            logger.warning(
                                "Groq model %s failed (%s); trying fallback model",
                                model_name,
                                exc,
                            )
                            model_failed = True
                            break

                        if exc.status_code in ROTATE_KEY_STATUSES:
                            logger.warning(
                                "Groq key failed with status %s; rotating key",
                                exc.status_code,
                            )
                            self._advance_key(keys, api_key)
                            should_backoff = exc.status_code == 429
                            continue

                        if exc.status_code in TRANSIENT_STATUSES:
                            logger.warning(
                                "Transient Groq error with model %s (%s); rotating key",
                                model_name,
                                exc,
                            )
                            self._advance_key(keys, api_key)
                            should_backoff = True
                            continue

                        raise

                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        last_error = exc
                        logger.warning("Groq transport error (%s); rotating key", exc)
                        self._advance_key(keys, api_key)
                        should_backoff = True
                        continue

                if attempt + 1 < retries and should_backoff:
                    self._sleep_before_retry(attempt, last_error)
                if model_failed:
                    break

        raise RuntimeError(
            "Groq completion failed after trying all configured keys and models. "
            f"Last error: {last_error}"
        )

    def _request(
        self,
        *,
        api_key: str,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{settings.GROQ_API_BASE_URL}/chat/completions"

        with httpx.Client(timeout=settings.GROQ_TIMEOUT_SECONDS) as client:
            resp = client.post(url, headers=headers, json=payload)

        if resp.status_code >= 400:
            raise GroqAPIError(
                self._error_message(resp),
                status_code=resp.status_code,
                retry_after=self._retry_after(resp),
            )

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GroqAPIError(
                f"Unexpected Groq response shape: {resp.text[:300]}",
                status_code=resp.status_code,
            ) from exc

        return LLMResponse(text=(text or "").strip(), model=model)

    def _models(self, model: Optional[str]) -> list[str]:
        ordered = [model or settings.GROQ_MODEL] + settings.GROQ_MODEL_FALLBACKS
        seen: set[str] = set()
        result: list[str] = []
        for item in ordered:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _ordered_keys(self, keys: list[str]) -> list[str]:
        if not keys:
            return []
        start = self._key_index % len(keys)
        return keys[start:] + keys[:start]

    def _remember_successful_key(self, keys: list[str], api_key: str) -> None:
        try:
            self._key_index = keys.index(api_key)
        except ValueError:
            self._key_index = 0

    def _advance_key(self, keys: list[str], api_key: str) -> None:
        try:
            self._key_index = (keys.index(api_key) + 1) % len(keys)
        except ValueError:
            self._key_index = 0

    def _sleep_before_retry(
        self,
        attempt: int,
        last_error: Optional[BaseException],
    ) -> None:
        retry_after = (
            last_error.retry_after
            if isinstance(last_error, GroqAPIError)
            else None
        )
        delay = retry_after or min(
            settings.GROQ_RETRY_BACKOFF_MAX_SEC,
            settings.GROQ_RETRY_BACKOFF_BASE_SEC * (2 ** attempt),
        )
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _retry_after(resp: httpx.Response) -> Optional[float]:
        raw = resp.headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _error_message(resp: httpx.Response) -> str:
        try:
            payload = resp.json()
        except ValueError:
            return f"Groq API error {resp.status_code}: {resp.text[:300]}"

        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or str(error)
        else:
            message = str(error or payload)
        return f"Groq API error {resp.status_code}: {message}"


_client: Optional[GroqLLMClient] = None


def get_llm_client() -> GroqLLMClient:
    global _client
    if _client is None:
        _client = GroqLLMClient()
    return _client
