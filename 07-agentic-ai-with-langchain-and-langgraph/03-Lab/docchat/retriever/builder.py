from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with embeddings."""
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_EMBEDDING_MODEL,
        )
        
    def build_hybrid_retriever(self, docs):
        """Build a retriever, preferring vector search and falling back to BM25."""
        try:
            # Create BM25 retriever
            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully.")

            try:
                # Create Chroma vector store
                vector_store = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    persist_directory=settings.CHROMA_DB_PATH
                )
                logger.info("Vector store created successfully.")

                # Create vector-based retriever
                vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
                logger.info("Vector retriever created successfully.")

                logger.info("Using vector retriever.")
                return vector_retriever
            except Exception as vector_error:
                logger.warning(f"Vector retrieval unavailable, using BM25 only: {vector_error}")
                return bm25
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise
