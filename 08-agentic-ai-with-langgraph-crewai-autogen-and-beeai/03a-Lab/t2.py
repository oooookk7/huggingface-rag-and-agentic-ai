import asyncio
import logging
import os
from beeai_framework.backend import ChatModel, ChatModelParameters, UserMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the chat model
async def basic_chat_example():
    print("Starting...")
    # Create a chat model instance (works with OpenAI, WatsonX, etc.)
    llm = ChatModel.from_name(
        "openai:Qwen/Qwen3.5-397B-A17B:novita",
        ChatModelParameters(temperature=0),
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"),
        base_url=os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1"),
    )
    
    # Create a conversation about something everyone finds interesting
    messages = [
        SystemMessage(content="You are a helpful AI assistant and creative writing expert."),
        UserMessage(content="Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")
    ]
    
    print("Generating response...")

    # Generate response using create() method
    response = await llm.create(messages=messages)
    
    print("User: Help me brainstorm a unique business idea for a food delivery service that doesn't exist yet.")
    print(f"Assistant: {response.get_text_content()}")
    
    return response

# Run the basic chat example
async def main() -> None:
    print("START")
    logging.getLogger('asyncio').setLevel(logging.CRITICAL) # Suppress unwanted warnings
    response = await basic_chat_example()
    print("END")

if __name__ == "__main__":
    asyncio.run(main())
