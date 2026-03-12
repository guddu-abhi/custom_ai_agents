from pydantic import BaseModel


class AgentsBaseResponse(BaseModel):
    response: str
    follow_up_needed: bool  = False
    agent_handed_off_to: str | None = None  # Track if this agent handed off to another agent
