"""Tests for the RAG pipeline."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, config
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


DATASET_PATH = DATA_DIR / "email_dataset.json"


@pytest.fixture
def dataset():
    if not DATASET_PATH.exists():
        pytest.skip("Dataset not found. Run generate_dataset.py first.")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def embedding_model():
    return EmbeddingModel()


class TestEmbeddings:
    def test_embed_text_returns_list(self, embedding_model):
        result = embedding_model.embed_text("Hello, this is a test email.")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    def test_embed_batch(self, embedding_model):
        texts = ["Hello world", "This is a test", "Email about meeting"]
        results = embedding_model.embed_batch(texts)
        assert len(results) == 3
        assert all(len(r) == len(results[0]) for r in results)

    def test_similar_texts_have_close_embeddings(self, embedding_model):
        import numpy as np
        e1 = np.array(embedding_model.embed_text("Schedule a meeting for tomorrow"))
        e2 = np.array(embedding_model.embed_text("Can we set up a meeting tomorrow?"))
        e3 = np.array(embedding_model.embed_text("The financial report shows losses"))

        sim_12 = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        sim_13 = np.dot(e1, e3) / (np.linalg.norm(e1) * np.linalg.norm(e3))

        assert sim_12 > sim_13, "Similar texts should have higher cosine similarity"


class TestVectorStore:
    def test_index_and_query(self, dataset, embedding_model):
        store = VectorStore(embedding_model)
        train_data = [d for d in dataset if d["split"] == "train"][:10]
        count = store.index_dataset(train_data)
        assert count == len(train_data)

        results = store.query("I need to schedule a meeting", k=3)
        assert len(results) <= 3
        assert all(r.similarity_score >= 0 for r in results)

    def test_query_returns_relevant_results(self, dataset, embedding_model):
        store = VectorStore(embedding_model)
        train_data = [d for d in dataset if d["split"] == "train"][:20]
        store.index_dataset(train_data)

        results = store.query("I need to reschedule our meeting", k=5)
        assert len(results) > 0
        assert results[0].similarity_score > 0.3

    def test_exclude_ids(self, dataset, embedding_model):
        store = VectorStore(embedding_model)
        train_data = [d for d in dataset if d["split"] == "train"][:10]
        store.index_dataset(train_data)

        exclude = [train_data[0]["id"]]
        results = store.query("test query", k=3, exclude_ids=exclude)
        result_ids = [r.id for r in results]
        assert exclude[0] not in result_ids


class TestPipelineIntegration:
    @pytest.mark.skipif(not config.google_api_key, reason="GOOGLE_API_KEY not set")
    def test_full_pipeline(self, dataset):
        from src.pipeline import Pipeline

        pipeline = Pipeline()
        pipeline.dataset = dataset
        pipeline.index_dataset(split="train")

        reply = pipeline.respond(
            from_addr="test@example.com",
            subject="Quick Question About Meeting",
            body="Can we move our 3pm meeting to 4pm tomorrow?",
        )

        assert reply.reply_text
        assert len(reply.reply_text) > 20
        assert reply.model == config.generation_model
        assert reply.latency_seconds > 0
        assert len(reply.retrieved_ids) > 0
