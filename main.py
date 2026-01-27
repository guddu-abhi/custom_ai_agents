from pydantic import BaseModel

from agents import Agent, Runner
from loguru import logger
# Customer Desk Agent prompt for triaging support requests.
CUSTOMER_DESK_PROMPT = (
    "You are a customer support operations agent. "
    "Your job is to triage incoming customer requests and assign them to the correct department: "
    "account, billing, incident, or policy. "
    "For each request, analyze the content and select the most appropriate department. "
    "Return a short summary explaining your reasoning and the chosen department."
)

class TriageResult(BaseModel):
    department: str  # One of: account, billing, incident, policy
    summary: str     # Short explanation of the triage decision

customer_desk_agent = Agent(
    name="CustomerDeskAgent",
    instructions=CUSTOMER_DESK_PROMPT,
    output_type=TriageResult,
    model="gpt-5-nano"
)

if __name__ == "__main__":
    # Example usage of the customer desk agent
    example_request = (
        "I need help updating my billing information for my account. "
        "Can you assist me with that?"
    )
    logger.info(example_request)
    #result = Runner.run_sync(customer_desk_agent, example_request)
    # print(result.final_output)