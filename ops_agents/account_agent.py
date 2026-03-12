from agents import Agent

from ops_agents.model.base_response import AgentsBaseResponse

# Account Agent prompt for handling account-related customer queries.
ACCOUNT_AGENT_PROMPT = (
    "You are an expert account support agent. "
    "Your job is to help customers with any account-related questions, such as login issues, profile updates, account status, or access problems. "
    "For each customer query, provide a clear, concise, and actionable response that resolves their account issue or guides them to the next step. "
    "If you need more information, ask clarifying questions. Always be polite and helpful."
    "If the customer replies with \"back to original agent\" transfer back to the triage agent."
)


class AccountAgentResponse(AgentsBaseResponse):
    pass


account_agent = Agent(
    name="AccountAgent",
    instructions=ACCOUNT_AGENT_PROMPT,
    output_type=AccountAgentResponse,
    model="gpt-5-nano",
)
