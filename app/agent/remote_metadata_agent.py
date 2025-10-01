from mas.schema import ModelOutput
from typing import List
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
structured_model = model.with_structured_output(ModelOutput)

def run(metadata: Dict[str, Any]):
    prompt=SystemMessage("""
        You are a cybersecurity expert specializing in phishing, with a particular focus on email contents. 
        Your task is to scrutinize the email URLs for any signs of fraud, urgency, or threats. Judge whether this email contain phishing intent. Provide a confidence score between 0 and 1 and a clear, concise explanation of your reasoning.""")

    content=str(metadata or {})
    return structured_model.invoke([sys, content])