from typing import TypedDict
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict, Literal
from datetime import datetime


class ModelOutput(BaseModel):
    score: float
    reasons: List[str]

class AgentOutput(BaseModel):
    agent: str
    score: float
    reasons: List[str]
    features: Dict[str, Any] = Field(default_factory=dict)


"""
Email Schema
"""
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

"""
Phone call
"""
class AudioClipInfo(BaseModel):
    """
    Optional metadata for an audio clip used by agents or UI.
    """
    path: str
    duration_sec: Optional[float] = None
    sampling_rate: Optional[int] = None
    format: Optional[str] = None  # e.g., 'wav', 'm4a'


class CallMetadata(BaseModel):
    """
    High-level call metadata (device/platform can fill what it knows).
    """
    caller_number: Optional[str] = None
    callee_number: Optional[str] = None
    direction: Optional[Literal["incoming", "outgoing"]] = None
    start_ts: Optional[datetime] = None
    duration_sec: Optional[int] = None
    presentation: Optional[Literal["allowed", "restricted", "unknown", "payphone"]] = None
    device_id: Optional[str] = None
    country_code: Optional[str] = None  # e.g., 'AU', 'US'


class AnalyzeCallRequest(BaseModel):
    """
    Request payload to analyze a phone call for phishing risk.
    - `audio_paths`: local paths to one or more voicemail/recorded audio files.
    - `metadata`: optional call metadata (numbers, direction, etc.).
    - `transcripts`: optional STT transcripts if already available; can be omitted when analyzing raw audio only.
    """
    call_id: Optional[str] = None
    audio_paths: List[str] = Field(default_factory=list)
    metadata: Optional[CallMetadata] = None
    transcripts: Optional[List[str]] = None


class PhoneGraphState(TypedDict, total=False):
    """
    State container passed among nodes in the phone-call graph.
    Mirrors the pattern used in your email graph but scoped to phone only.
    """
    # Inputs
    request: AnalyzeCallRequest
    audio_paths: List[str]
    transcripts: List[str]
    metadata: CallMetadata

    # Extracted/derived
    _text_: str
    _urls_: List[str]
    audio_clips_info: List[AudioClipInfo]

    # Agent outputs
    audio_output: AgentOutput
    text_output: AgentOutput
    url_output: AgentOutput
    metadata_output: AgentOutput

    # Aggregation
    local_outputs: List[AgentOutput]
    remote_outputs: List[AgentOutput]
    decision: FinalDecision