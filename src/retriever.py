from typing import List, Optional
from dataclasses import dataclass
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore, EmailPair


@dataclass
class RetrievedContext:
    email_pairs: List[EmailPair]
    query_text: str

    def format_for_prompt(self) -> str:
        examples = []
        for i, pair in enumerate(self.email_pairs, 1):
            examples.append(
                f"### Example {i} (similarity: {pair.similarity_score:.2f})\n"
                f"**Incoming Email:**\n"
                f"From: {pair.incoming_from}\n"
                f"Subject: {pair.incoming_subject}\n"
                f"Body: {pair.incoming_body}\n\n"
                f"**Reply:**\n"
                f"Subject: {pair.reply_subject}\n"
                f"Body: {pair.reply_body}"
            )
        return "\n\n---\n\n".join(examples)


class Retriever:
    def __init__(self, vector_store: VectorStore = None, embedding_model: EmbeddingModel = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore(self.embedding_model)

    def retrieve(self, email_text: str, k: int = None, exclude_ids: Optional[List[str]] = None) -> RetrievedContext:
        pairs = self.vector_store.query(email_text, k=k, exclude_ids=exclude_ids)
        return RetrievedContext(email_pairs=pairs, query_text=email_text)
