from typing import Dict, List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from config.settings import settings


class VerificationAgent:
    def __init__(self):
        """Initialize the verification agent with an OpenAI-compatible chat model."""
        self.model = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            max_tokens=300,
        )

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        return f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

        **Instructions:**
        - Verify the following answer against the provided context.
        - Check for:
        1. Direct/indirect factual support (YES/NO)
        2. Unsupported claims (list any if present)
        3. Contradictions (list any if present)
        4. Relevance to the question (YES/NO)
        - Provide additional details or explanations where relevant.
        - Respond in the exact format specified below without adding any unrelated information.

        **Format:**
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]

        **Answer:** {answer}
        **Context:**
        {context}

        **Respond ONLY with the above format.**
        """

    def parse_verification_response(self, response_text: str) -> Dict:
        try:
            lines = response_text.split("\n")
            verification = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().capitalize()
                    value = value.strip()
                    if key in {"Supported", "Unsupported claims", "Contradictions", "Relevant", "Additional details"}:
                        if key in {"Unsupported claims", "Contradictions"}:
                            if value.startswith("[") and value.endswith("]"):
                                items = value[1:-1].split(",")
                                items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                                verification[key] = items
                            else:
                                verification[key] = []
                        elif key == "Additional details":
                            verification[key] = value
                        else:
                            verification[key] = value.upper()

            for key in ["Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"]:
                if key not in verification:
                    if key in {"Unsupported Claims", "Contradictions"}:
                        verification[key] = []
                    elif key == "Additional Details":
                        verification[key] = ""
                    else:
                        verification[key] = "NO"

            return verification
        except Exception:
            return None

    def format_verification_report(self, verification: Dict) -> str:
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        report = f"**Supported:** {supported}\n"
        report += f"**Unsupported Claims:** {', '.join(unsupported_claims) if unsupported_claims else 'None'}\n"
        report += f"**Contradictions:** {', '.join(contradictions) if contradictions else 'None'}\n"
        report += f"**Relevant:** {relevant}\n"
        report += f"**Additional Details:** {additional_details if additional_details else 'None'}\n"
        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(answer, context)

        try:
            llm_response = self.model.invoke(prompt).content.strip()
        except Exception as e:
            raise RuntimeError("Failed to verify answer due to a model error.") from e

        sanitized_response = self.sanitize_response(llm_response) if llm_response else ""
        if not sanitized_response:
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model.",
            }
        else:
            verification_report = self.parse_verification_response(sanitized_response)
            if verification_report is None:
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response.",
                }

        return {
            "verification_report": self.format_verification_report(verification_report),
            "context_used": context,
        }
