import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from src.config import config, CHROMA_DIR
from src.embeddings import EmbeddingModel


@dataclass
class EmailPair:
    id: str
    category: str
    scenario: str
    incoming_subject: str
    incoming_body: str
    incoming_from: str
    reply_subject: str
    reply_body: str
    metadata: Dict[str, Any]
    similarity_score: float = 0.0


class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=config.chroma_collection,
            metadata={"hnsw:space": "cosine"}
        )

    def index_dataset(self, dataset: List[Dict[str, Any]]) -> int:
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        texts_to_embed = []
        for item in dataset:
            email = item["incoming_email"]
            text = f"Subject: {email['subject']}\n\n{email['body']}"
            texts_to_embed.append(text)

        all_embeddings = self.embedding_model.embed_batch(texts_to_embed)

        for i, item in enumerate(dataset):
            email = item["incoming_email"]
            reply = item["expected_reply"]
            text = texts_to_embed[i]

            ids.append(item["id"])
            documents.append(text)
            embeddings.append(all_embeddings[i])
            metadatas.append({
                "category": item["category"],
                "scenario": item["scenario"],
                "reply_subject": reply["subject"],
                "reply_body": reply["body"],
                "incoming_from": email.get("from", ""),
                "incoming_to": email.get("to", ""),
                "tone": item.get("metadata", {}).get("tone", ""),
                "intent": item.get("metadata", {}).get("intent", ""),
                "key_actions": json.dumps(item.get("metadata", {}).get("key_actions", [])),
                "complexity": item.get("metadata", {}).get("complexity", ""),
                "split": item.get("split", "train"),
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    def query(self, text: str, k: int = None, exclude_ids: Optional[List[str]] = None) -> List[EmailPair]:
        k = k or config.top_k_retrieval
        query_k = k + len(exclude_ids) if exclude_ids else k

        embedding = self.embedding_model.embed_text(text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(query_k, self.collection.count()),
        )

        pairs = []
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            if exclude_ids and doc_id in exclude_ids:
                continue

            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0.0
            similarity = 1.0 - distance

            pairs.append(EmailPair(
                id=doc_id,
                category=meta.get("category", ""),
                scenario=meta.get("scenario", ""),
                incoming_subject=results["documents"][0][i].split("\n\n")[0].replace("Subject: ", ""),
                incoming_body="\n\n".join(results["documents"][0][i].split("\n\n")[1:]),
                incoming_from=meta.get("incoming_from", ""),
                reply_subject=meta.get("reply_subject", ""),
                reply_body=meta.get("reply_body", ""),
                metadata={
                    "tone": meta.get("tone", ""),
                    "intent": meta.get("intent", ""),
                    "key_actions": json.loads(meta.get("key_actions", "[]")),
                    "complexity": meta.get("complexity", ""),
                },
                similarity_score=similarity,
            ))
            if len(pairs) >= k:
                break

        return pairs

    def count(self) -> int:
        return self.collection.count()
