import asyncio
import logging
import json
import re
from pydantic import BaseModel, Field
from typing import List
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Define a structured output for business planning
class BusinessPlan(BaseModel):
    """A comprehensive business plan structure."""
    business_name: str = Field(description="Catchy name for the business")
    elevator_pitch: str = Field(description="30-second description of the business")
    target_market: str = Field(description="Primary target audience")
    unique_value_proposition: str = Field(description="What makes this business special")
    revenue_streams: List[str] = Field(description="Ways the business will make money")
    startup_costs: str = Field(description="Estimated initial investment needed")
    key_success_factors: List[str] = Field(description="Critical elements for success")


def _extract_first_json(text: str) -> dict:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("No valid JSON object found in model output.")


def _coerce_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        # Support comma/newline/bullet separated strings.
        parts = re.split(r",|\n|;|•|- ", value)
        return [p.strip() for p in parts if p.strip()]
    return []


async def structured_output_example():
    llm = ChatModel.from_name(
        "openai:Qwen/Qwen3.5-397B-A17B:novita",
        ChatModelParameters(temperature=0),
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"),
        base_url=os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1"),
    )

    messages = [
        SystemMessage(
            content=(
                "You are an expert business consultant. "
                "Return ONLY valid JSON with these exact keys: "
                "business_name, elevator_pitch, target_market, unique_value_proposition, "
                "revenue_streams, startup_costs, key_success_factors. "
                "For revenue_streams and key_success_factors, return JSON arrays."
            )
        ),
        UserMessage(
            content=(
                "Create a business plan for a mobile app that helps people find and book "
                "unique local experiences in their city."
            )
        ),
    ]

    response = await llm.create(messages=messages)
    raw_text = response.get_text_content()
    payload = _extract_first_json(raw_text)
    payload["revenue_streams"] = _coerce_list(payload.get("revenue_streams"))
    payload["key_success_factors"] = _coerce_list(payload.get("key_success_factors"))
    plan = BusinessPlan.model_validate(payload)

    print("User: Create a business plan for a mobile app that helps people find and book unique local experiences in their city.")
    print("\n🚀 AI-Generated Business Plan:")
    print(f"💡 Business Name: {plan.business_name}")
    print(f"🎯 Elevator Pitch: {plan.elevator_pitch}")
    print(f"👥 Target Market: {plan.target_market}")
    print(f"⭐ Unique Value Proposition: {plan.unique_value_proposition}")
    print(f"💰 Revenue Streams: {', '.join(plan.revenue_streams)}")
    print(f"💵 Startup Costs: {plan.startup_costs}")
    print(f"🔑 Key Success Factors:")
    for factor in plan.key_success_factors:
        print(f"  - {factor}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL) # Suppress unwanted warnings
    await structured_output_example()

if __name__ == "__main__":
    asyncio.run(main())
