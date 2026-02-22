# Import necessary packages
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import gradio as gr
import os

# Load .env from parent folder (02-build-rag-applications/.env)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Model settings
model_id = "zai-org/GLM-5"
model_provider = "novita"

# Set necessary parameters
parameters = {
    "temperature": 0.8,  # This controls randomness/creativity
    "top_p": 0.9,
}

# Wrap model with Hugging Face endpoint + chat interface
hf_endpoint = HuggingFaceEndpoint(
    repo_id=model_id,
    provider=model_provider,
    task="conversational",
    temperature=parameters["temperature"],
    top_p=parameters["top_p"],
    max_new_tokens=256,
    model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
)
hf_llm = ChatHuggingFace(llm=hf_endpoint)

# Function to generate a response from the model
def generate_response(prompt_txt):
    generated_response = hf_llm.invoke(prompt_txt)
    return generated_response.content if hasattr(generated_response, "content") else str(generated_response)

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    flagging_mode="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Hugging Face GLM-5 Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

if __name__ == "__main__":
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError(
            "Missing Hugging Face token. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) in 02-build-rag-applications/.env."
        )
    chat_application.launch(server_name="127.0.0.1", server_port=7860)
