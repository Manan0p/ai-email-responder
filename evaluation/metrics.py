import json
import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer, util


@dataclass
class MetricScores:
    rouge_l: float
    bert_score: float
    sentence_similarity: float
    key_action_coverage: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rouge_l": round(self.rouge_l, 4),
            "bert_score": round(self.bert_score, 4),
            "sentence_similarity": round(self.sentence_similarity, 4),
            "key_action_coverage": round(self.key_action_coverage, 4),
        }


class AutomatedMetrics:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self._st_model = None
        self._embedding_model_name = embedding_model_name

    @property
    def st_model(self) -> SentenceTransformer:
        if self._st_model is None:
            self._st_model = SentenceTransformer(self._embedding_model_name)
        return self._st_model

    def rouge_l(self, generated: str, reference: str) -> float:
        scores = self.rouge.score(reference, generated)
        return scores["rougeL"].fmeasure

    def compute_bert_score(self, generated: str, reference: str) -> float:
        P, R, F1 = bert_score_fn(
            [generated], [reference],
            lang="en",
            verbose=False,
            rescale_with_baseline=True,
        )
        return max(0.0, F1.item())

    def sentence_similarity(self, generated: str, reference: str) -> float:
        embeddings = self.st_model.encode([generated, reference], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return float(similarity.item())

    def key_action_coverage(self, generated: str, key_actions: List[str]) -> float:
        if not key_actions:
            return 1.0

        generated_lower = generated.lower()
        covered = 0

        action_keywords = {
            "apologize": ["sorry", "apologize", "apolog", "regret"],
            "address_concerns": ["concern", "worry", "issue", "understand your", "address"],
        }

        stopwords = {"the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with", "new"}

        for action in key_actions:
            action_lower = action.lower()
            keywords = action_keywords.get(action_lower.replace(" ", "_"))
            if keywords is None:
                words = [w for w in re.findall(r"[a-z]+", action_lower) if w not in stopwords and len(w) > 3]
                keywords = words or [action_lower]
            if any(kw in generated_lower for kw in keywords):
                covered += 1

        return covered / len(key_actions)

    def compute_all(self, generated: str, reference: str, key_actions: List[str] = None) -> MetricScores:
        return MetricScores(
            rouge_l=self.rouge_l(generated, reference),
            bert_score=self.compute_bert_score(generated, reference),
            sentence_similarity=self.sentence_similarity(generated, reference),
            key_action_coverage=self.key_action_coverage(generated, key_actions or []),
        )

    def compute_batch(self, generated_list: List[str], reference_list: List[str], key_actions_list: List[List[str]] = None) -> List[MetricScores]:
        if key_actions_list is None:
            key_actions_list = [[] for _ in generated_list]

        results = []
        for gen, ref, actions in zip(generated_list, reference_list, key_actions_list):
            results.append(self.compute_all(gen, ref, actions))
        return results
