from typing import TypedDict
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ModelOutput(BaseModel):
    score: float
    reasons: List[str]

class AgentOutput(BaseModel):
    agent: str
    score: float
    reasons: List[str]
    features: Dict[str, Any] = Field(default_factory=dict)

class FinalDecision(BaseModel):
    risk_label: str
    risk_prob: float
    top_reasons: List[str]
    agent_outputs: List[AgentOutput]


class EmailGraphState(TypedDict, total=False):
    # inputs
    message_id: str
    headers_raw: str
    subject: str
    body_html: Optional[str]
    body_text: Optional[str]
    metadata: Dict[str, Any]

    # derived/intermediate
    _text_: str
    _urls_: List[str]
    local_outputs: List[AgentOutput]
    local_score: float

    # agent outputs
    text_output: AgentOutput
    url_output: AgentOutput
    meta_output: AgentOutput
    text_remote_output: List[AgentOutput]
    url_remote_output: List[AgentOutput]
    meta_remote_output: List[AgentOutput]

    # final
    decision: FinalDecision

class AnalyzeEmailRequest(BaseModel):
    message_id: str
    headers_raw: str
    subject: str
    body_html: Optional[str] = None
    body_text: Optional[str] = None
    received_at: Optional[str] = None
