from agents import Agent
from pydantic import BaseModel

# Billing Agent prompt for handling billing-related customer queries.
BILLING_AGENT_PROMPT = (
    "You are an expert billing support agent. "
    "Your job is to help customers with any billing-related questions, such as invoices, payments, refunds, charges, or subscription issues. "
    "For each customer query, provide a clear, concise, and actionable response that resolves their billing issue or guides them to the next step. "
    "If you need more information, ask clarifying questions. Always be polite and helpful."
)

class BillingAgentResponse(BaseModel):
    resolution: str  # The answer or next step for the customer's billing question
    follow_up_needed: bool  # True if more info is needed from the customer

billing_agent = Agent(
    name="BillingAgent",
    instructions=BILLING_AGENT_PROMPT,
    output_type=BillingAgentResponse,
    model="gpt-5-nano"
)
