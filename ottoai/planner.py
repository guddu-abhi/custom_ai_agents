from agents import Agent

from domain.models.search_plan import ConversationalSearchPlan, SearchPlan
from ottoai.config import settings

PLANNER_INSTRUCTIONS = (
    "You are the query planner for a product search assistant. "
    "Given the user's question, produce a SearchPlan:\n"
    "1. `query`: rewrite the question into a concise keyword search query suitable "
    "for product search. Drop filler words but KEEP the salient product terms, "
    "including the product type / category the user mentions (e.g. 'noise "
    "cancelling headphones', 'usb-c laptop charger'). This query feeds a semantic "
    "vector search, so descriptive product words improve results — do not strip "
    "the category out of the query.\n"
    "2. `filters`: extract brand, price_max (USD), min_rating (0-5), and "
    "min_reviews (integer) ONLY when the user states them explicitly. Leave any "
    "unstated filter as null. Do not invent filters. There is no category filter "
    "— express category in `query`.\n"
    "   - min_reviews: set when the user asks for popular / well-reviewed / "
    "trusted / best-selling products (e.g. 'popular' or 'well-reviewed' -> ~100, "
    "'some reviews' -> ~50).\n"
    "   - Pairing rule: average rating alone is unreliable (a 5.0 from 2 reviews "
    "beats a 4.6 from 10k). So whenever you set `min_rating`, also set "
    "`min_reviews` to at least 50 so the rating floor is backed by enough reviews."
)

planner_agent = Agent(
    name="OttoPlanner",
    instructions=PLANNER_INSTRUCTIONS,
    model=settings.planner_model,
    output_type=SearchPlan,
)


CONVERSATIONAL_PLANNER_INSTRUCTIONS = (
    "You are the query planner for a MULTI-TURN product search assistant. You are "
    "given the conversation so far, the filters currently active, the products "
    "already shown to the user, and the user's latest message. Produce a "
    "ConversationalSearchPlan:\n"
    "1. `query`: a STANDALONE keyword search query for the latest message, "
    "resolved against the conversation (expand pronouns/ellipsis like 'cheaper "
    "ones' or 'what about Sony' into a full query). Keep the product type / "
    "category in the query — there is no category filter.\n"
    "2. `filters`: a DELTA — only the filters this turn ADDS or CHANGES "
    "(brand, price_max USD, min_rating 0-5, min_reviews int). Leave unchanged "
    "filters null; they carry over automatically. Pairing rule: whenever you set "
    "`min_rating`, also set `min_reviews` to at least 50. Set min_reviews ~100 for "
    "'popular'/'well-reviewed'.\n"
    "3. `reset_filters`: true only if the user wants to drop ALL prior constraints "
    "(e.g. 'forget the budget', 'show me anything').\n"
    "4. `needs_retrieval`: false ONLY when the latest message can be answered from "
    "the products ALREADY SHOWN (e.g. 'is the second one waterproof?', 'compare "
    "the first two', 'tell me more about the Sony'). true when the user wants "
    "different/new/more products or changes any filter. When false, `query` and "
    "`filters` may stay empty/null."
)

conversational_planner_agent = Agent(
    name="OttoConversationalPlanner",
    instructions=CONVERSATIONAL_PLANNER_INSTRUCTIONS,
    model=settings.planner_model,
    output_type=ConversationalSearchPlan,
)
