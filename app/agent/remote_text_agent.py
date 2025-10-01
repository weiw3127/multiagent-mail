# email_remote_text_agent.py
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
#from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from app.schema import ModelOutput

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
structured = model.with_structured_output(ModelOutput)

def run(subject: str, content: str) -> ModelOutput:
    sys = SystemMessage(
        content=(
            "You are a cybersecurity expert specializing in phishing emails. "
            "Given the subject and body, return a JSON with {score:[0..1], reasons:[...]} "
            "where score≈likelihood of phishing."
        )
    )
    human = HumanMessage(content=f"Subject:\n{subject}\n\nBody:\n{content}")
    
    return structured.invoke([sys, human])
