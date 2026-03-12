from pydantic import BaseModel


class ConversationResponse(BaseModel):
    agent: str
    response: str
