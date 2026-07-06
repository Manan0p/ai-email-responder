import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.config import config, DATA_DIR
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator, GeneratedReply


@dataclass
class PipelineResult:
    email_id: str
    incoming_email: Dict[str, str]
    generated_reply: GeneratedReply
    expected_reply: Optional[Dict[str, str]] = None


class Pipeline:
    def __init__(self):
        self.dataset: List[Dict[str, Any]] = []
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(self.embedding_model)
        self.retriever = Retriever(self.vector_store, self.embedding_model)
        self.generator = Generator(self.retriever)
        self._indexed = False

    def load_dataset(self, path: str = None) -> List[Dict[str, Any]]:
        path = path or config.dataset_path
        with open(path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        return self.dataset

    def index_dataset(self, split: str = None) -> int:
        data = self.dataset
        if split:
            data = [d for d in self.dataset if d.get("split") == split]
        count = self.vector_store.index_dataset(data)
        self._indexed = True
        return count

    def respond(self, email_text: str = None, from_addr: str = "", subject: str = "", body: str = "", exclude_ids: Optional[List[str]] = None) -> GeneratedReply:
        if not self._indexed:
            raise RuntimeError("Dataset not indexed. Call load_dataset() and index_dataset() first.")

        if email_text and not body:
            lines = email_text.strip().split("\n")
            subject_line = ""
            body_lines = []
            for line in lines:
                if line.lower().startswith("subject:") and not subject_line:
                    subject_line = line.split(":", 1)[1].strip()
                else:
                    body_lines.append(line)
            subject = subject or subject_line
            body = "\n".join(body_lines).strip()

        return self.generator.generate(from_addr, subject, body, exclude_ids=exclude_ids)

    def evaluate_test_set(self, limit: int = None) -> List[PipelineResult]:
        test_items = [d for d in self.dataset if d.get("split") == "test"]
        if limit:
            test_items = test_items[:limit]
        results = []

        for item in test_items:
            email = item["incoming_email"]
            reply = self.generator.generate(
                from_addr=email.get("from", ""),
                subject=email.get("subject", ""),
                body=email.get("body", ""),
                exclude_ids=[item["id"]],
            )
            results.append(PipelineResult(
                email_id=item["id"],
                incoming_email=email,
                generated_reply=reply,
                expected_reply=item.get("expected_reply"),
            ))

        return results

    def get_train_set(self) -> List[Dict[str, Any]]:
        return [d for d in self.dataset if d.get("split") == "train"]

    def get_test_set(self) -> List[Dict[str, Any]]:
        return [d for d in self.dataset if d.get("split") == "test"]
