"""Client for the Bifrost OpenAI-compatible chat completions API.

Credentials and the model name come from config.py. API keys are never
printed, logged, or included in error messages.
"""

from __future__ import annotations

import requests

from config import BIFROST_API_KEY, BIFROST_BASE_URL, BIFROST_MODEL

# Generation can take longer than a typical HTTP call.
_REQUEST_TIMEOUT_SECONDS = 310


class BifrostClient:
    """Call the local Bifrost gateway and return generated text."""

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:
        """Send a system/user chat completion and return the assistant text."""
        url = f"{BIFROST_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {BIFROST_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": BIFROST_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                "Bifrost request timed out before a response was received."
            ) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError("Could not connect to the Bifrost gateway.") from exc
        except requests.RequestException as exc:
            raise RuntimeError("Bifrost request failed before a response was received.") from exc

        if response.status_code >= 400:
            raise RuntimeError(f"Bifrost returned HTTP {response.status_code}.")

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Bifrost returned a response that is not valid JSON.") from exc

        return _extract_content(data)


def _extract_content(data: object) -> str:
    """Return choices[0].message.content, or raise if the shape is unexpected."""
    if not isinstance(data, dict):
        raise RuntimeError(
            "Bifrost response did not include choices[0].message.content."
        )

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            "Bifrost response did not include choices[0].message.content."
        )

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError(
            "Bifrost response did not include choices[0].message.content."
        )

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(
            "Bifrost response did not include choices[0].message.content."
        )
    return content
