from typing import Dict, List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from config.settings import settings


class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with an OpenAI-compatible chat model."""
        self.model = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL,
            temperature=0.3,
            max_tokens=300,
        )

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        return f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.

        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """

    def generate(self, question: str, documents: List[Document]) -> Dict:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(question, context)

        try:
            llm_response = self.model.invoke(prompt).content.strip()
        except Exception as e:
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        return {
            "draft_answer": draft_answer,
            "context_used": context,
        }
