from dataclasses import dataclass, field
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from app.schema import (
    AgentOutput, AnalyzeCallRequest, PhoneGraphState, FinalDecision
)

from app.agent import local_audio_agent

PHONE_THRESHOLD = 0.50

def _decide(outputs: List[AgentOutput]) -> FinalDecision:
    outs = [o for o in outputs if o is not None]
    if not outs:
        return FinalDecision(
            risk_label="safe",
            risk_prob=0.0,
            top_reasons=["No signals available"],
            agent_outputs=[],
        )

    scores = [o.score for o in outs]
    risk = float(sum(scores) / max(len(scores), 1))
    label = "safe"
    if risk >= 0.8:
        label = "phishing"
    elif risk >= PHONE_THRESHOLD:
        label = "suspicious"

    reasons: List[str] = []
    for o in outs:
        for r in o.reasons:
            if len(reasons) < 5 and r not in reasons:
                reasons.append(r)

    return FinalDecision(
        risk_label=label,
        risk_prob=risk,
        top_reasons=reasons,
        agent_outputs=outs,
    )

@dataclass
class PhoneOrchestrator:
    graph: Any = field(init=False)

    def __post_init__(self):
        g = StateGraph(PhoneGraphState)

        def start_node(state: PhoneGraphState) -> Dict[str, Any]:
            req: AnalyzeCallRequest = state.get("request")  # type: ignore
            audio_paths = list(req.audio_paths or [])
            return {"audio_paths": audio_paths}

        def audio_node(state: PhoneGraphState) -> Dict[str, Any]:
            out: AgentOutput = local_audio_agent.run(state.get("audio_paths", []))
            return {"audio_output": out}

        def local_aggregate_node(state: PhoneGraphState) -> Dict[str, Any]:
            outs: List[AgentOutput] = [state.get("audio_output")]
            decision: FinalDecision = _decide([o for o in outs if o])
            return {"local_outputs": outs, "decision": decision}

        def finalize_node(state: PhoneGraphState) -> Dict[str, Any]:
            # placeholder
            return {}

        g.add_node("start", start_node)
        g.add_node("audio", audio_node)
        g.add_node("local_aggregate", local_aggregate_node)
        g.add_node("finalize", finalize_node)

        g.set_entry_point("start")
        g.add_edge("start", "audio")
        g.add_edge("audio", "local_aggregate")
        g.add_edge("local_aggregate", "finalize")
        g.add_edge("finalize", END)

        self.graph = g.compile()

    def analyze(self, req: AnalyzeCallRequest) -> FinalDecision:
        initial: Dict[str, Any] = {"request": req}
        state = self.graph.invoke(initial)
        if not isinstance(state, dict):
            raise RuntimeError("LangGraph returned unexpected state type")
        decision = state.get("decision")
        if isinstance(decision, FinalDecision):
            return decision
        raise RuntimeError("Graph execution did not produce a FinalDecision")
