import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

load_dotenv()

# Pydantic models: these are the "guard rails" around the LLM

DELIM = "<<<META>>>"
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)


class Settings(BaseModel):
    """App/model settings. Temperature is hard-capped so nobody can raise it."""

    model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    temperature: float = Field(0.2, ge=0.0, le=0.3)
    # "low" = fastest first token; "minimal" is NOT supported by 3.8 Flash
    thinking_level: Literal["low", "medium", "high"] = "low"
    max_tokens: int = Field(2048, ge=64, le=8192)  # includes thinking tokens
    history_window: int = Field(10, ge=2, le=40)  # messages sent as context


class UserMessage(BaseModel):
    """Validated user input."""

    text: str = Field(min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def clean(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message is empty.")
        return v


class ReplyMeta(BaseModel):
    """Structured metadata the model must append after its answer."""

    model_config = ConfigDict(extra="forbid")

    in_scope: bool
    confidence: Literal["high", "medium", "low"]
    memory_updates: list[str] = Field(default_factory=list, max_length=3)
    follow_ups: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("memory_updates", "follow_ups")
    @classmethod
    def short_items(cls, items: list[str]) -> list[str]:
        return [i.strip()[:120] for i in items if i.strip()]

# Prompt

BASE_SYSTEM_PROMPT = f"""You are "Sage", a friendly, patient study assistant for beginners learning \
AI, machine learning, Python and programming.

SCOPE
- Answer only questions about AI/ML, Python, programming, and study/learning advice.
- If a request is outside this scope, decline briefly and politely and steer back. Set in_scope to false.

ACCURACY RULES (very important)
- Never invent facts, function names, library APIs, links, statistics or citations.
- If you are not sure, say "I'm not sure" and say what the user should verify in official docs.
- Do not guess about events or versions you cannot know. Prefer short, correct answers over long, uncertain ones.
- Only use facts from this conversation or well-established knowledge.

STYLE
- Warm, encouraging, concise. Explain simply first, then add detail if useful.
- Use Markdown. Put all code in fenced code blocks with a language tag.

OUTPUT FORMAT (strict)
1. Write your answer in Markdown.
2. Then, on a new line, write exactly {DELIM} followed by ONE JSON object and nothing else \
(no code fences):
{{"in_scope": true|false, "confidence": "high"|"medium"|"low", \
"memory_updates": [], "follow_ups": []}}
- confidence: how sure you are that the answer is factually correct.
- memory_updates: up to 3 durable facts the user EXPLICITLY stated about themselves \
(name, skill level, goals). Otherwise [].
- follow_ups: up to 3 short suggested next questions. May be [].
"""


def build_system_prompt(memory: list[str]) -> str:
    if not memory:
        return BASE_SYSTEM_PROMPT
    facts = "\n".join(f"- {m}" for m in memory)
    return (
        BASE_SYSTEM_PROMPT
        + "\nKNOWN FACTS ABOUT THE USER (from earlier chats; use naturally, don't repeat unprompted):\n"
        + facts
        + "\n"
    )

# Helpers

def visible_part(raw: str) -> str:
    """Return the text to show the user: everything before the META delimiter.
    Also hides a partially-streamed delimiter such as '<<<ME'."""
    if DELIM in raw:
        return raw.split(DELIM, 1)[0].rstrip()
    for i in range(len(DELIM) - 1, 0, -1):
        if raw.endswith(DELIM[:i]):
            return raw[:-i]
    return raw


def parse_meta(raw: str) -> ReplyMeta | None:
    if DELIM not in raw:
        return None
    tail = raw.split(DELIM, 1)[1].strip()
    tail = tail.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return ReplyMeta.model_validate_json(tail)
    except ValidationError:
        return None


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        if code in (401, 403):
            return "Authentication failed. Check your GEMINI_API_KEY."
        if code == 404:
            return "Model not found. Check the GEMINI_MODEL value in your .env file."
        if code == 429:
            return "Rate limit / quota reached. Please wait a moment and try again."
        if code == 400:
            return f"The API rejected the request: {getattr(exc, 'message', exc)}"
        if code and code >= 500:
            return "Gemini is temporarily unavailable. Please try again shortly."
        return f"API error ({code}): {getattr(exc, 'message', exc)}"
    if isinstance(exc, httpx.TimeoutException):
        return "The request timed out. Please try again."
    if isinstance(exc, httpx.TransportError):
        return "Could not reach the API. Check your internet connection."
    return f"Unexpected error: {exc}"


@st.cache_resource
def get_client() -> genai.Client:
    # Reads GEMINI_API_KEY (or GOOGLE_API_KEY). SDK retries transient errors (429/5xx).
    return genai.Client(
        http_options=types.HttpOptions(
            timeout=60_000,  # ms
            retry_options=types.HttpRetryOptions(attempts=3),
        )
    )


def to_contents(messages: list[dict], window: int) -> list[types.Content]:
    """Sliding context window: last N messages, always starting with a user turn.
    Gemini uses the role name 'model' instead of 'assistant'."""
    recent = messages[-window:]
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    return [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in recent
    ]


def build_config(settings: "Settings", system: str, use_temperature: bool) -> types.GenerateContentConfig:
    kwargs = dict(
        system_instruction=system,
        max_output_tokens=settings.max_tokens,
        thinking_config=types.ThinkingConfig(thinking_level=settings.thinking_level),
    )
    if use_temperature:
        kwargs["temperature"] = settings.temperature
    return types.GenerateContentConfig(**kwargs)


def stream_gemini(settings: "Settings", system: str, contents: list, on_text) -> tuple[str, str | None]:
    """Stream a reply. Calls on_text(chunk) per chunk. Returns (raw_text, finish_reason).

    Gemini 3.x docs/tutorials disagree on whether a custom temperature is accepted. We try the
    low temperature first; if the API rejects it with a 400 mentioning 'temperature', we retry
    once without it and remember that for the session (thinking_level='low' then keeps it focused).
    """
    use_temp = st.session_state.get("use_temperature", True)
    for attempt in range(2):
        raw, finish, got_text = "", None, False
        try:
            stream = get_client().models.generate_content_stream(
                model=settings.model,
                contents=contents,
                config=build_config(settings, system, use_temp),
            )
            for chunk in stream:
                if chunk.candidates and chunk.candidates[0].finish_reason:
                    finish = str(chunk.candidates[0].finish_reason)
                text = chunk.text or ""
                if text:
                    got_text = True
                    raw += text
                    on_text(text, raw)
            if not got_text:
                raise RuntimeError(
                    f"The model returned no text (finish reason: {finish or 'unknown'}). "
                    "It may have been blocked by a safety filter. Try rephrasing."
                )
            return raw, finish
        except errors.ClientError as exc:
            msg = str(getattr(exc, "message", exc)).lower()
            if attempt == 0 and use_temp and getattr(exc, "code", None) == 400 and "temperature" in msg:
                use_temp = False
                st.session_state.use_temperature = False
                continue
            raise
    raise RuntimeError("Unreachable")


# ---- persistence ---------------------------------------------------------- #


def save_chat() -> None:
    if not st.session_state.messages:
        return
    first_user = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "Chat")
    data = {
        "id": st.session_state.chat_id,
        "title": first_user[:40],
        "updated": datetime.now().isoformat(timespec="seconds"),
        "messages": st.session_state.messages,
    }
    (HISTORY_DIR / f"{st.session_state.chat_id}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_chats() -> list[dict]:
    chats = []
    for p in HISTORY_DIR.glob("*.json"):
        if p.name.startswith("_"):  # skip internal files such as _memory.json
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "id" in data and "messages" in data:
            chats.append(data)
    return sorted(chats, key=lambda c: c.get("updated", ""), reverse=True)


def load_memory() -> list[str]:
    p = HISTORY_DIR / "_memory.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    except (json.JSONDecodeError, OSError):
        return []


def save_memory() -> None:
    (HISTORY_DIR / "_memory.json").write_text(
        json.dumps(st.session_state.memory, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def new_chat() -> None:
    st.session_state.chat_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []


def render_meta(meta: dict | None, ttft: float | None) -> None:
    if meta is None:
        st.caption("Could not verify this answer's metadata. Double-check important details.")
        return
    if not meta["in_scope"]:
        st.caption("Out of scope: I only help with AI, Python and programming topics.")
    elif meta["confidence"] == "low":
        st.warning("Low confidence: please verify this with official documentation.", icon="⚠️")
    elif meta["confidence"] == "medium":
        st.caption("Medium confidence")
    if meta["follow_ups"]:
        st.caption("You can try asking: " + "  |  ".join(meta["follow_ups"]))
    if ttft is not None:
        st.caption(f"First token in {ttft:.2f}s")

# UI

st.set_page_config(page_title="Sage: AI Study Chatbot :)")

if "chat_id" not in st.session_state:
    new_chat()
if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

# Sidebar
with st.sidebar:
    st.title("Sage")
    st.caption("Your AI & Python study buddy")

    temperature = st.slider("Temperature (capped for accuracy)", 0.0, 0.3, 0.2, 0.05)
    window = st.slider("Context window (messages)", 2, 40, 10, 2)
    try:
        settings = Settings(temperature=temperature, history_window=window)
    except ValidationError as e:
        st.error(f"Invalid settings: {e}")
        st.stop()
    st.caption(f"Model: `{settings.model}` · thinking: `{settings.thinking_level}`")
    if st.session_state.get("use_temperature") is False:
        st.caption("ℹ️ This model rejected a custom temperature, so it's disabled. "
                   "Low thinking level is used to keep answers focused.")

    c1, c2 = st.columns(2)
    if c1.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()
    if c2.button("Clear chat", use_container_width=True):
        old = HISTORY_DIR / f"{st.session_state.chat_id}.json"
        old.unlink(missing_ok=True)
        st.session_state.messages = []
        st.rerun()


    st.divider()
    st.subheader("History")
    for chat in list_chats()[:10]:
        if st.button(chat["title"] or "Chat", key=f"load_{chat['id']}", use_container_width=True):
            st.session_state.chat_id = chat["id"]
            st.session_state.messages = chat["messages"]
            st.rerun()

st.title("Sage: AI Study Buddy!")

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    st.error("GEMINI_API_KEY is not set. Copy `.env.example` to `.env` and add your key "
             "(get one at https://aistudio.google.com/apikey).")
    st.stop()

# Render existing conversation
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and "meta" in m:
            render_meta(m["meta"], m.get("ttft"))

if not st.session_state.messages:
    st.info("Hi, I'm Sage! Your personal AI Chatbot! You can ask me about Python, AI or programming! :)")

# Input + streaming response
if prompt := st.chat_input("Ask me anything about AI or Python..."):
    try:
        user_msg = UserMessage(text=prompt)
    except ValidationError:
        st.error("Please enter a message between 1 and 2000 characters.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_msg.text})
    with st.chat_message("user"):
        st.markdown(user_msg.text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(":) _Thinking..._")  # loading state until first token
        timing = {"ttft": None}
        t0 = time.perf_counter()

        def on_text(_chunk: str, raw_so_far: str) -> None:
            if timing["ttft"] is None:
                timing["ttft"] = time.perf_counter() - t0
            placeholder.markdown(visible_part(raw_so_far) + "▌")

        try:
            raw, finish = stream_gemini(
                settings,
                build_system_prompt(st.session_state.memory),
                to_contents(st.session_state.messages, settings.history_window),
                on_text,
            )
        except Exception as exc:  # noqa: BLE001 - mapped to friendly message
            placeholder.empty()
            st.error(friendly_error(exc))
            st.session_state.messages.pop()  # let the user resend the same question
            st.stop()

        ttft = timing["ttft"]
        answer = visible_part(raw).strip() or "_(No response received. Please try again.)_"
        placeholder.markdown(answer)
        if finish and "MAX_TOKENS" in finish:
            st.caption("Answer was cut off (max tokens reached) :(. Ask me to continue.")

        meta = parse_meta(raw)
        render_meta(meta.model_dump() if meta else None, ttft)

        # Update long-term memory from validated metadata only
        if meta:
            for fact in meta.memory_updates:
                if fact not in st.session_state.memory:
                    st.session_state.memory.append(fact)
            st.session_state.memory = st.session_state.memory[-20:]
            save_memory()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "meta": meta.model_dump() if meta else None,
            "ttft": ttft,
        }
    )
    save_chat()
    if meta and meta.memory_updates:
        st.rerun()  # refresh sidebar so new memory shows immediately
