"""
chat_engine.py

Handles all communication with the Anthropic Claude API.

Why this file exists separately from app.py:
    Keeping AI/SDK logic out of the UI layer means:
    1. It can be unit-tested without spinning up Streamlit.
    2. The UI framework (Streamlit today) can be swapped for FastAPI,
       a CLI, or a Discord bot later without touching this file at all.
    3. Anyone reading the repo can find "how it talks to Claude" in one
       obvious place.
"""

import os
from typing import Dict, List

from anthropic import Anthropic, APIConnectionError, APIError, RateLimitError
from dotenv import load_dotenv

# Load key-value pairs from a local .env file into the process environment.
# Must run before we try to read ANTHROPIC_API_KEY below.
load_dotenv()

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
MODEL_NAME: str = "claude-sonnet-4-6"  # Best speed/intelligence balance for this use case
MAX_TOKENS: int = 1024                 # Caps reply length -> predictable cost per call
SYSTEM_PROMPT: str = (
    "You are a helpful, concise AI assistant built as a portfolio project "
    "for a Generative AI internship. Keep answers clear and to the point."
)


def get_client() -> Anthropic:
    """
    Build and return an authenticated Anthropic SDK client.

    Returns:
        Anthropic: A ready-to-use client instance.

    Raises:
        ValueError: If the ANTHROPIC_API_KEY environment variable is not set.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
            "paste in your key from https://console.anthropic.com/settings/keys"
        )
    return Anthropic(api_key=api_key)


def get_claude_response(conversation_history: List[Dict[str, str]]) -> str:
    """
    Send the full conversation history to Claude and return its text reply.

    Args:
        conversation_history: The ENTIRE transcript so far, as a list of
            {"role": "user" | "assistant", "content": "..."} dicts.
            Claude's API is stateless, so this whole list must be resent
            on every single call — that's how "memory" is simulated.

    Returns:
        str: The text content of Claude's reply.

    Raises:
        ValueError: If conversation_history is empty.
        RuntimeError: If the API call fails (network issue, rate limit,
            bad key, etc.), wrapped in a clear message instead of leaking
            a raw SDK traceback up to the UI layer.
    """
    if not conversation_history:
        raise ValueError("conversation_history cannot be empty.")

    client = get_client()

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=conversation_history,
        )
    except RateLimitError as exc:
        raise RuntimeError(
            "Claude API rate limit reached. Wait a moment and try again."
        ) from exc
    except APIConnectionError as exc:
        raise RuntimeError(
            "Could not reach the Claude API. Check your internet connection."
        ) from exc
    except APIError as exc:
        raise RuntimeError(f"Claude API returned an error: {exc}") from exc

    # response.content is a list of content blocks. For a plain-text
    # reply (no tool calls, no images), the first block holds the text.
    return response.content[0].text
