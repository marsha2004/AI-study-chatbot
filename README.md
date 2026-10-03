# Sage: AI Study Chatbot

A streaming chatbot with a defined role (beginner-friendly AI & Python tutor), built with
**Streamlit**, the **Gemini API (`gemini-3.8-flash`)** and **Pydantic**.

## Features

| Requirement | Implementation |
|---|---|
| User input / AI response | `st.chat_input` + streamed assistant reply |
| API integration | `google-genai` SDK (`generate_content_stream`) with `gemini-3.8-flash`, key loaded from `.env` |
| Loading state | "Thinking..." placeholder until the first token arrives |
| Error handling | Typed API exceptions mapped to friendly messages; input validation |
| System prompt | "Sage" persona, scope limits, accuracy rules, strict output format |
| Streaming + TTFT | Tokens rendered live; time-to-first-token is measured and shown |
| Anti-hallucination | Low temperature (hard-capped at 0.3 by Pydantic, with automatic fallback if the model rejects it) + `thinking_level="low"`, "say I'm not sure" rules, Pydantic-validated reply metadata with confidence + scope flags |
| Conversation history | Chats saved to `chat_history/`, reload from sidebar |
| Clear chat / New chat | Sidebar buttons |
| Context / memory | Sliding context window + persistent facts about the user |
| Markdown | Rendered natively, code blocks supported |

## Setup

```bash
git clone <your-repo-url>
cd ai-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your GEMINI_API_KEY (https://aistudio.google.com/apikey)
streamlit run app.py
```

## Note on temperature (Gemini 3.x)

Google's docs and third-party guides disagree on whether Gemini 3.x accepts a custom `temperature`
(Google recommends the default of 1.0; at least one guide says a custom value returns a 400 error).
The app therefore **tries `temperature=0.2` first**. If the API rejects it with a 400 mentioning
temperature, the app retries once without it, remembers that for the session, and shows a note in the
sidebar. In both cases `thinking_level="low"` is used, which also gives the fastest first token.

## Try it

- `My name is Sam and I'm a beginner` → saved to memory (sidebar)
- `Explain list comprehensions with an example` → Markdown + code, streamed
- `Who won the cricket match yesterday?` → politely declined (out of scope)
- Disconnect the internet / use a bad key → friendly error message

## Project structure

```
app.py            # whole application
requirements.txt
.env.example
chat_history/     # created at runtime (git-ignored)
screenshots/      # add your screenshots here
REPORT.md         # Day 2 & 3 report
```
