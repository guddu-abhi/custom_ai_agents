from pydantic import BaseModel

from agents import Agent
from pydantic import BaseModel

# Account Agent prompt for handling account-related customer queries.
ACCOUNT_AGENT_PROMPT = (
    "You are an expert account support agent. "
    "Your job is to help customers with any account-related questions, such as login issues, profile updates, account status, or access problems. "
    "For each customer query, provide a clear, concise, and actionable response that resolves their account issue or guides them to the next step. "
    "If you need more information, ask clarifying questions. Always be polite and helpful."
)

class AccountAgentResponse(BaseModel):
    resolution: str  # The answer or next step for the customer's account question
    follow_up_needed: bool  # True if more info is needed from the customer

account_agent = Agent(
    name="AccountAgent",
    instructions=ACCOUNT_AGENT_PROMPT,
    output_type=AccountAgentResponse,
    model="gpt-5-nano"
)
