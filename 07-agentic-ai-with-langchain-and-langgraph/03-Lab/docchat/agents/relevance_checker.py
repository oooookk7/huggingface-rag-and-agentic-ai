from langchain_openai import ChatOpenAI
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RelevanceChecker:
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL,
            temperature=0,
            max_tokens=10,
        )

    def check(self, question: str, retriever, k=3) -> str:
        top_docs = retriever.invoke(question)
        if not top_docs:
            return "NO_MATCH"

        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.

        **Instructions:**
        - Classify how well the document content addresses the user's question.
        - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
        - Do not include any additional text or explanation.

        **Labels:**
        1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
        2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
        3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

        **Important:** If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

        **Question:** {question}
        **Passages:** {document_content}

        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """

        try:
            llm_response = self.model.invoke(prompt).content.strip().upper()
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return "NO_MATCH"

        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        return llm_response if llm_response in valid_labels else "NO_MATCH"
