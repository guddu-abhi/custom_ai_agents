from agents import Agent
from pydantic import BaseModel

# Policy Agent prompt for handling policy-related customer queries.
POLICY_AGENT_PROMPT = (
    "You are an expert policy support agent. "
    "Your job is to help customers with any policy-related questions, such as company policies, terms of service, privacy, compliance, or eligibility. "
    "For each customer query, provide a clear, concise, and actionable response that addresses their policy question or guides them to the next step. "
    "If you need more information, ask clarifying questions. Always be polite and helpful."
)

class PolicyAgentResponse(BaseModel):
    resolution: str  # The answer or next step for the customer's policy question
    follow_up_needed: bool  # True if more info is needed from the customer

policy_agent = Agent(
    name="PolicyAgent",
    instructions=POLICY_AGENT_PROMPT,
    output_type=PolicyAgentResponse,
    model="gpt-5-nano"
)
