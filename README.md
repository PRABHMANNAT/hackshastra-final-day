# SWASTH AI Dashboard

Premium tactical mental-health monitoring dashboard built with React + Vite.

## Setup

1. Copy `.env.example` to `.env` if you do not already have one.
2. Add the four env values used by the app:
   `VITE_OPENAI_API_KEY`, `VITE_OPENAI_CHAT_MODEL`, `VITE_SPEECH_API_KEY`, `VITE_SPEECH_MODEL`.
3. Install dependencies:

```bash
npm install
```

4. Start the app:

```bash
npm run dev
```

5. Open the local Vite URL in Chrome or Safari for microphone support.

## Notes

- The React client talks to local Vite middleware routes so the API keys stay off the browser bundle.
- `/api/chat-stream` proxies OpenAI Chat Completions with streaming enabled.
- `/api/transcribe` uses `VITE_SPEECH_API_KEY` and OpenAI transcription for the mic flow.
- `/api/speech` uses `VITE_SPEECH_API_KEY` + `VITE_SPEECH_MODEL` for spoken assistant replies.
