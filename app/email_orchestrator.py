from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from langgraph.graph import END, StateGraph
from app.agent import (
    local_text_agent,
    local_url_agent,
    remote_metadata_agent,
    remote_text_agent,
    remote_url_agent,
)
from app.constant import EMAIL_THRESHOLD
from app.schema import AgentOutput, EmailGraphState, FinalDecision, AnalyzeEmailRequest
from app.util import decide, extract_urls, html_to_text


@dataclass
class EmailOrchestrator:
    graph: Any = field(init=False)

    def __post_init__(self) -> None: 
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Any:
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

        def remote_start_node(state: EmailGraphState) -> Dict[str, Any]:
            return {
                "text_remote_output": state.get("text_remote_output", []),
                "url_remote_output": state.get("url_remote_output", []),
                "meta_remote_output": state.get("meta_remote_output", []),
            }
        
        def remote_metadata_node(state: EmailGraphState) -> Dict[str, Any]:
            meta_payload: Dict[str, Any] = {
                "headers_raw": state.get("headers_raw", ""),
                "message_id": state.get("message_id", ""),
            }
            model_response = remote_metadata_agent.run(meta_payload)
            output = AgentOutput(agent="email_metadata_agent_remote", score=model_response.score, reasons=model_response.reasons)
            lst = list(state.get("meta_remote_output", []))
            lst.append(output)
            return {"meta_remote_output": lst}
        
        def remote_text_node(state: EmailGraphState) -> Dict[str, Any]:
            model_response =  remote_text_agent.run(state.get("subject", ""), state.get("_text_", ""))
            output = AgentOutput(agent="email_text_agent_remote", score=model_response.score, reasons=model_response.reasons)
            lst = list(state.get("text_remote_output", []))
            lst.append(output)
            return {"text_remote_output": lst}

        def remote_url_node(state: EmailGraphState) -> Dict[str, Any]:
            model_response =  remote_url_agent.run(state.get("_urls_", []))
            output = AgentOutput(agent="email_url_agent_remote", score=model_response.score, reasons=model_response.reasons)
            lst = list(state.get("url_remote_output", []))
            lst.append(output)
            return {"url_remote_output": lst}

        def remote_aggregate_node(state: EmailGraphState) -> Dict[str, Any]:
            local_outs: List[AgentOutput] = state.get("local_outputs", [])
            rem_outs: List[AgentOutput] = (
                state.get("text_remote_output", [])
                + state.get("url_remote_output", [])
                + state.get("meta_remote_output", [])
            )
            final_decision: FinalDecision = decide(local_outs + rem_outs)
            return {"decision": final_decision}
        
        # langGraph nodes
        g.add_node("start", start_node)
        g.add_node("text", text_node)
        g.add_node("url", url_node)
        g.add_node("local_aggregate", local_aggregate_node)
        g.add_node("finalize", finalize_local_node)

        g.add_node("remote_start", remote_start_node)
        g.add_node("remote_text", remote_text_node)
        g.add_node("remote_url", remote_url_node)
        g.add_node("remote_metadata", remote_metadata_node)
        g.add_node("remote_aggregate", remote_aggregate_node)

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
                "accept": "finalize",
                "escalate": "remote_start",
            },
        )

        g.add_edge("remote_start", "remote_text")
        g.add_edge("remote_text", "remote_url")
        g.add_edge("remote_url", "remote_metadata")
        g.add_edge("remote_metadata", "remote_aggregate")

        g.add_edge("remote_aggregate", "finalize")
        g.add_edge("finalize", END)

        return g.compile()

    def analyze(self, req: AnalyzeEmailRequest) -> FinalDecision:
        initial: EmailGraphState = {
            "message_id": req.message_id,
            "headers_raw": req.headers_raw or "",
            "subject": req.subject or "",
            "body_html": req.body_html,
            "body_text": req.body_text,
        }

        state = self.graph.invoke(initial)
        if not isinstance(state, dict):
            raise RuntimeError("LangGraph returned unexpected state type")

        decision = state.get("decision")
        if isinstance(decision, FinalDecision):
            return decision

        raise RuntimeError("Graph execution did not produce a FinalDecision")
