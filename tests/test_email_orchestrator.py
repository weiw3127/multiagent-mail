import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schema import AgentOutput, AnalyzeEmailRequest, ModelOutput


def _stub_module(name: str, run_fn):
    module = types.ModuleType(name)
    module.run = run_fn
    return module


@pytest.mark.parametrize("remote_scores", [(0.95, 0.9, 0.85)])
def test_email_orchestrator_executes_remote_branches(monkeypatch, remote_scores):
    call_sequence = []

    def local_text(subject: str, text: str) -> AgentOutput:
        call_sequence.append("local_text")
        return AgentOutput(agent="local_text", score=0.2, reasons=["local text"], features={})

    def local_url(urls) -> AgentOutput:
        call_sequence.append("local_url")
        return AgentOutput(agent="local_url", score=0.2, reasons=["local url"], features={})

    remote_text_score, remote_url_score, remote_meta_score = remote_scores

    def remote_text(subject: str, text: str) -> ModelOutput:
        call_sequence.append("remote_text")
        return ModelOutput(score=remote_text_score, reasons=["remote text"])

    def remote_url(urls) -> ModelOutput:
        call_sequence.append("remote_url")
        return ModelOutput(score=remote_url_score, reasons=["remote url"])

    def remote_meta(metadata) -> ModelOutput:
        call_sequence.append("remote_metadata")
        return ModelOutput(score=remote_meta_score, reasons=["remote metadata"])

    stub_map = {
        "app.agent.local_text_agent": local_text,
        "app.agent.local_url_agent": local_url,
        "app.agent.remote_text_agent": remote_text,
        "app.agent.remote_url_agent": remote_url,
        "app.agent.remote_metadata_agent": remote_meta,
    }

    for module_name in stub_map:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    monkeypatch.delitem(sys.modules, "app.email_orchestrator", raising=False)

    for module_name, fn in stub_map.items():
        monkeypatch.setitem(sys.modules, module_name, _stub_module(module_name, fn))

    email_orchestrator = importlib.import_module("app.email_orchestrator")

    orchestrator = email_orchestrator.EmailOrchestrator()

    request = AnalyzeEmailRequest(
        message_id="test",
        headers_raw="",
        subject="Important update",
        body_html=None,
        body_text="Please visit https://phish.example to review",
    )

    decision = orchestrator.analyze(request)

    assert decision.risk_prob == pytest.approx(
        round((0.2 + 0.2 + sum(remote_scores)) / 5, 3)
    )
    assert decision.risk_label == "MEDIUM"
    assert len(decision.agent_outputs) == 5
    assert "remote metadata" in decision.top_reasons

    remote_calls = [entry for entry in call_sequence if entry.startswith("remote")]
    assert remote_calls == ["remote_text", "remote_url", "remote_metadata"]
