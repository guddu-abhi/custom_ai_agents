# Post Order Support Agent
from agents import Agent
from ops_agents.model.base_response import AgentsBaseResponse


SYSTEM_PROMPT = (
    "You are the Post Order Support Agent for a consumer electronics store. "
    "Assist customers with any issues after their purchase, such as refunds, warranty, or store policies. "
    "Be empathetic and helpful, making sure the customer feels supported. "
    "Answer questions about product returns, warranty claims, and store policies. "
    "Guide the customer through the process and resolve their concerns efficiently."
)

class PostOrderSupportAgentResponse(AgentsBaseResponse):
    pass


post_order_support_agent = Agent(
    name="PostOrderSupportAgent",
    instructions=SYSTEM_PROMPT,
    output_type=PostOrderSupportAgentResponse,
    model="gpt-5-nano",
)
