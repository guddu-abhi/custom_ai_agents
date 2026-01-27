from agents import Agent
from pydantic import BaseModel

# Incident Agent prompt for handling incident-related customer queries.
INCIDENT_AGENT_PROMPT = (
    "You are an expert incident support agent. "
    "Your job is to help customers with any incident-related questions, such as service outages, technical issues, security incidents, or urgent disruptions. "
    "For each customer query, provide a clear, concise, and actionable response that addresses their incident or guides them to the next step. "
    "If you need more information, ask clarifying questions. Always be polite and helpful."
)

class IncidentAgentResponse(BaseModel):
    resolution: str  # The answer or next step for the customer's incident question
    follow_up_needed: bool  # True if more info is needed from the customer

incident_agent = Agent(
    name="IncidentAgent",
    instructions=INCIDENT_AGENT_PROMPT,
    output_type=IncidentAgentResponse,
    model="gpt-5-nano"
)
