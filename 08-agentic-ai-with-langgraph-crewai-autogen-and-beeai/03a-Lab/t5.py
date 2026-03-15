import asyncio
import logging
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel, ChatModelParameters
import os
from dotenv import load_dotenv

load_dotenv()

async def minimal_tracked_agent_example():
    """
    Minimal RequirementAgent
    """
    llm = ChatModel.from_name(
        "openai:Qwen/Qwen3.5-397B-A17B:novita",
        ChatModelParameters(temperature=0),
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"),
        base_url=os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1"),
    )

    # CONSISTENT SYSTEM PROMPT (used in all examples)
    SYSTEM_INSTRUCTIONS = """You are an expert cybersecurity analyst specializing in threat assessment and risk analysis.

Your methodology:
1. Analyze the threat landscape systematically
2. Research authoritative sources when available
3. Provide comprehensive risk assessment with actionable recommendations
4. Focus on practical, implementable security measures"""

    # Minimal RequirementAgent
    minimal_agent = RequirementAgent(
        llm=llm,
        tools=[],  # No tools yet
        memory=UnconstrainedMemory(),
        instructions=SYSTEM_INSTRUCTIONS
    )
    
    # CONSISTENT QUERY (used in all examples)
    ANALYSIS_QUERY = """Analyze the cybersecurity risks of quantum computing for financial institutions. 
    What are the main threats, timeline for concern, and recommended preparation strategies?"""
    
    result = await minimal_agent.run(ANALYSIS_QUERY)
    print(f"\n💬 Pure LLM Analysis:\n{result.answer.text}")

async def main() -> None:
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    await minimal_tracked_agent_example()

if __name__ == "__main__":
    asyncio.run(main())
