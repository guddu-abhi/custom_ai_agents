"""
Agent Registry - Central mapping of all available agents by name.
This allows dynamic agent lookup during conversation handoffs.
"""

from ops_agents.customer_triage_agent import customer_desk_agent
from ops_agents.account_agent import account_agent
from ops_agents.billing_agent import billing_agent
from ops_agents.product_advisor_agent import product_advisor_agent
from ops_agents.post_order_support_agent import post_order_support_agent  # Placeholder for future agent
# Agent registry mapping agent names to agent instances
AGENT_REGISTRY = {
    "CustomerDeskAgent": customer_desk_agent,
    "AccountAgent": account_agent,
    "BillingAgent": billing_agent,
    "ProductAdvisorAgent": product_advisor_agent,
    "PostOrderSupportAgent": post_order_support_agent,  # Placeholder for future agent
}


def get_agent_by_name(agent_name: str):
    """
    Retrieve an agent instance by name.
    Defaults to customer_desk_agent if agent not found.
    """
    return AGENT_REGISTRY.get(agent_name)
