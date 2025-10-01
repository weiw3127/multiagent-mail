# mailMAS

mailMAS is a multi-agent email security service that scores inbound messages for phishing risk. It combines open-source Hugging Face classifiers with Google Gemini models orchestrated through a LangGraph state machine to deliver fast local heuristics and deeper remote analysis when the risk score warrants escalation.

## Key Features
- Multi-stage phishing detection that escalates from local transformers to Gemini 2.5 evaluations when confidence is low.
- URL, text, and metadata analyzers that contribute structured evidence toward a final decision.
- FastAPI service with JSON endpoints for easy integration into mail pipelines or security tooling.
- Ready-to-run smoke test and pytest suite to validate the orchestration logic.

## Project Layout
- `app/main.py` – FastAPI application exposing `/health` and `/email/analyze`.
- `app/email_orchestrator.py` – LangGraph workflow that coordinates agents and aggregates their scores.
- `app/agent/` – Individual local (Hugging Face) and remote (Gemini) agents for text, URLs, and metadata.
- `app/schema.py` – Pydantic models for requests, agent outputs, and final decisions.
- `scripts/smoke_test_email.py` – CLI utility that exercises the live API.
- `tests/` – Pytest coverage for the orchestrator graph wiring.

## Requirements
- Python 3.10+
- Google AI Studio (Gemini) API key with access to `gemini-2.5-flash`
- Internet access for downloading the Hugging Face models on first run

Install dependencies with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
- **Google API key**: export `GOOGLE_API_KEY` before starting the app. If the variable is absent, the service prompts for it on startup (`app/main.py` and `config.py`).
- **Risk threshold**: adjust `EMAIL_THRESHOLD` in `app/constant.py` to tune when the graph escalates from local to remote agents.

## Running the API
Start the FastAPI server with uvicorn:

```bash
uvicorn app.main:app --reload
```

Endpoints:
- `GET /health` → returns `{ "ok": true }`
- `POST /email/analyze` → accepts `AnalyzeEmailRequest` JSON and returns a `FinalDecision`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/email/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "demo-001",
    "headers_raw": "Received: ...",
    "subject": "Important: verify your account",
    "body_text": "Visit https://secure.example to keep access",
    "body_html": null
  }'
```

Example response (abridged):

```json
{
  "risk_label": "MEDIUM",
  "risk_prob": 0.48,
  "top_reasons": [
    "Classifier vote: phishing",
    "URL model red flag https://secure.example"
  ],
  "agent_outputs": [
    {"agent": "text", "score": 0.62, "reasons": ["Classifier vote: phishing"]},
    {"agent": "url", "score": 0.72, "reasons": ["URL model red flag https://secure.example"]}
  ]
}
```

## Smoke Test
With the server running, exercise both endpoints:

```bash
python scripts/smoke_test_email.py
```

The script pings `/health` and submits a sample email, printing both responses.

## Running the Test Suite
Execute the pytest suite to verify the LangGraph wiring and decision logic:

```bash
pytest
```

## Extending mailMAS
- Add new agents by implementing a `run(...)` function under `app/agent/` and attaching it in `EmailOrchestrator._build_graph`.
- Swap the default Gemini model by changing the `ChatGoogleGenerativeAI` configuration in the remote agents.
- Replace Hugging Face classifiers with internal models by editing `local_text_agent.py` and `local_url_agent.py`.


