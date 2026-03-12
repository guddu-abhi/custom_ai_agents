from agents import Agent
from pydantic import BaseModel

from ops_agents.model.base_response import AgentsBaseResponse
BILLING_AGENT_PROMPT = (
    "You are the Billing Agent for a consumer electronics store. "
    "Help customers place orders for products like iPhones, Android phones, and headphones. "
    "Assist with billing questions, payments, and order processing. "
    "Be clear, polite, and guide the customer through the purchase or resolve any billing issues. "
    "If the customer needs to return to the triage agent, set the flag 'agent_handed_off_to' to 'customer_desk_agent' in your response."
)


class BillingAgentResponse(AgentsBaseResponse):
    pass


billing_agent = Agent(
    name="BillingAgent",
    instructions=BILLING_AGENT_PROMPT,
    output_type=BillingAgentResponse,
)
