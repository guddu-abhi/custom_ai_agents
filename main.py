from agents import Agent, Runner, TResponseInputItem
from sub_agents.account_agent import account_agent
from sub_agents.billing_agent import billing_agent
from sub_agents.incident_agent import incident_agent
from sub_agents.policy_agent import policy_agent
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents import SQLiteSession
from loguru import logger
import sys
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

# Configure logger for better visibility
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Customer Desk Agent prompt for triaging and routing support requests
CUSTOMER_DESK_PROMPT = (
    "You are a friendly customer support operations agent. "
    "Your job is to understand customer requests and route them to the appropriate specialist department: "
    "account, billing, incident, or policy. "
    "\n\n"
    "When a customer first contacts you:\n"
    "1. Greet them warmly and ask how you can help\n"
    "2. Listen to their request carefully\n"
    "3. If the request clearly matches a department, immediately hand off to the appropriate specialist agent. Do not ask unnecessary follow-up questions.\n"
    "4. Only ask clarifying questions if the request is ambiguous and you cannot determine the correct department.\n"
    "5. Once you understand their needs, summarize briefly and hand off to the appropriate specialist.\n"
    "\n"
    "Be conversational, friendly, and helpful. Avoid repeating clarifying questions. Do not loop or delay the handoff if the department is clear."
    "\n"
    "Examples of direct handoff:\n"
    "- 'I need help with my bill' → Billing Agent\n"
    "- 'My account is locked' → Account Agent\n"
    "- 'There is a technical issue' → Incident Agent\n"
    "- 'What is your refund policy?' → Policy Agent\n"
    "\n"
    "If the user includes the word 'override' in their query, hand off immediately without any clarifying questions."
    "\n"
    "Available agents:\n"
    "- Account Agent: for account-related issues\n"
    "- Billing Agent: for billing and payment questions\n"
    "- Incident Agent: for reporting technical incidents\n"
    "- Policy Agent: for questions about company policies\n"
)

# Create a session instance that will persist across runs
# session_id = "conversation_123"
# session = SQLiteSession(session_id)
# Create a session instance with a session ID.
# This example uses an in-memory SQLite database.
# The `create_tables=True` flag is useful for development and testing.
session = SQLAlchemySession.from_url(
    "conversation_234",
    url="sqlite+aiosqlite:///./agent_memory.db",
    create_tables=True,
)

# Main customer desk agent - acts as the triage/routing layer
customer_desk_agent = Agent(
    name="CustomerDeskAgent",
    instructions=prompt_with_handoff_instructions(CUSTOMER_DESK_PROMPT),
    model="gpt-5-nano",
    handoffs=[account_agent, billing_agent, incident_agent, policy_agent]
)



def run_interactive_chat():
    """
    Interactive chat loop for customer support.
    Allows continuous conversation with the customer desk agent and its handoffs.
    """
    logger.info("=" * 60)
    logger.info("Customer Support Chat - Interactive Mode")
    logger.info("=" * 60)
    logger.info("Type 'quit' or 'exit' to end the conversation")
    logger.info("=" * 60)
    
    print("\n👋 Welcome to Customer Support! How can I help you today?\n")
    
    # Maintain conversation history across all agents by accumulating input items.
    # This ensures that when handoffs occur, all agents can see the full conversation.
    input_items: list[TResponseInputItem] = []
    
    # Track the currently active agent (starts with customer desk agent)
    current_agent = customer_desk_agent
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
                print("\n👋 Conversation ended.\n")
                break

            # IMPORTANT: create fresh input_items every turn
            input_items = [
                {"role": "user", "content": user_input}
            ]

            result = Runner.run_sync(
                current_agent,
                input_items,
                session=session
            )

            previous_agent = current_agent
            current_agent = result.last_agent or current_agent

            if previous_agent != current_agent:
                logger.success(
                    f"Handoff: {previous_agent.name} -> {current_agent.name}"
                )

            # Display response
            if result.final_output:
                print(f"\nAgent: {result.final_output}\n")
            else:
                for msg in reversed(result.messages or []):
                    if msg.role == "assistant" and msg.content:
                        print(f"\nAgent: {msg.content}\n")
                        break

        except KeyboardInterrupt:
            print("\n👋 Conversation ended.\n")
            break


if __name__ == "__main__":
    run_interactive_chat()