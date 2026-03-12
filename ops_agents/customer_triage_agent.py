from agents import Agent, Runner, TResponseInputItem, handoff
from ops_agents import product_advisor_agent
from ops_agents.account_agent import account_agent
from ops_agents.billing_agent import billing_agent
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from ops_agents.model.base_response import AgentsBaseResponse
from utils.log_utils import get_logger
from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

logger = get_logger()

CUSTOMER_TRIAGE_PROMPT = f"""
You are the Customer Triage Agent for a consumer electronics store.
Your ONLY job is to determine the correct specialist agent and immediately hand off.
You are a router, not a problem solver.

Routing Rules:
- If the request is about product information, comparisons, features, recommendations, or advice → IMMEDIATELY hand off to Product Advisor Agent.
- If the request is about purchasing, placing an order, payments, invoices, or billing questions → IMMEDIATELY hand off to Billing Agent.
- If the request is about issues after purchase such as refunds, warranty, returns, damaged items, or store policy → IMMEDIATELY hand off to Post Order Support Agent.

Do NOT answer product questions yourself.
Do NOT provide product details.
Do NOT partially answer before handing off.
If routing is clear, hand off immediately without additional commentary.
Only ask clarifying questions if and only if the user intent is ambiguous.

Examples:
- 'Can you tell me about iPhones?' → Product Advisor Agent (immediate handoff)
- 'Compare iPhone and Samsung' → Product Advisor Agent (immediate handoff)
- 'I want to buy headphones' → Billing Agent (immediate handoff)
- 'I was charged twice' → Billing Agent (immediate handoff)
- 'My phone stopped working' → Post Order Support Agent (immediate handoff)
- 'What is your refund policy?' → Post Order Support Agent (immediate handoff)

once you decide what agent to handoff to, set the 'agent_handed_off_to' field in your response to the name of the agent you are handing off to (e.g. 'ProductAdvisorAgent', 'BillingAgent', 'PostOrderSupportAgent').
And also set internal handoff metadata so that the handoff is triggered immediately by the system.
"""

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
    instructions=prompt_with_handoff_instructions(CUSTOMER_TRIAGE_PROMPT),
    model="gpt-5-nano",
    handoffs=[product_advisor_agent],
    output_type=AgentsBaseResponse,
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
            input_items = [{"role": "user", "content": user_input}]

            result = Runner.run_sync(current_agent, input_items, session=session)

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
