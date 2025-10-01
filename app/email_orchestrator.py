from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from langgraph.graph import StateGraph, END
from app.schema import EmailGraphState, FinalDecision, AgentOutput

from app.agent import (
    local_text_agent,
    local_url_agent,
)
from app.util import decide, html_to_text, extract_urls  
from app.constant import EMAIL_THRESHOLD

@dataclass
class EmailOrchestrator():
    graph: StateGraph = field(init=False)

    def __post_init__(self) -> None: 
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph: 
        g = StateGraph(EmailGraphState)

        def start_node(state: EmailGraphState) -> Dict[str, Any]:
            html = state.get("body_html")
            text = state.get("body_text") or (html_to_text(html) if html else "")
            urls = extract_urls(html or "", text or "")
            return {"_text_": text, "_urls_": urls}

        # Local Model
        def text_node(state: EmailGraphState) -> Dict[str, Any]:
            out: AgentOutput = local_text_agent.run(state.get("subject", ""), state.get("_text_", ""))
            return {"text_output": out}

        def url_node(state: EmailGraphState) -> Dict[str, Any]:
            out: AgentOutput = local_url_agent.run(state.get("_urls_", []))
            return {"url_output": out}

        def local_aggregate_node(state: EmailGraphState) -> Dict[str, Any]:
            outs: List[AgentOutput] = [state.get("text_output"), state.get("url_output")]
            outs = [o for o in outs if o is not None]
            decision: FinalDecision = decide(outs)
            return {"local_outputs": outs, "decision": decision, "local_score": decision.risk_prob}
        
        def conditional_edge(state: EmailGraphState) -> Literal["escalate", "accept"]:
            s = float(state.get("local_score", 0.0))
            return "escalate" if s < float(EMAIL_THRESHOLD) else "accept"

        def finalize_local_node(state: EmailGraphState) -> Dict[str, Any]:
            return {"decision": state["decision"]}
        
        # langGraph nodes
        g.add_node("start", start_node)
        g.add_node("text", text_node)
        g.add_node("url", url_node)
        g.add_node("local_aggregate", local_aggregate_node)
        g.add_node("finalize_local", finalize_local_node)

        #langGraph edges
        g.set_entry_point("start")
        g.add_edge("start", "text")
        g.add_edge("start", "url")
        g.add_edge("text", "local_aggregate")
        g.add_edge("url", "local_aggregate")

        g.add_conditional_edges(
            "local_aggregate",
            conditional_edge,
            {
                "accept": "finalize_local",
                "escalate": "remote_start",
            },
        )

        return g