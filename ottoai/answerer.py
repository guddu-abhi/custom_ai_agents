from agents import Agent

from generation.core.prompt import SYSTEM_PROMPT
from ottoai.config import settings

answerer_agent = Agent(
    name="OttoAnswerer",
    instructions=SYSTEM_PROMPT,
    model=settings.answerer_model,
)
