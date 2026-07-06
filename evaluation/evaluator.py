from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from evaluation.metrics import AutomatedMetrics, MetricScores
from evaluation.llm_judge import LLMJudge, JudgeScore


COMPOSITE_WEIGHTS = {
    "rouge_l": 0.10,
    "bert_score": 0.20,
    "sentence_similarity": 0.15,
    "key_action_coverage": 0.15,
    "llm_judge": 0.40,
}


@dataclass
class EvaluationResult:
    email_id: str
    category: str
    scenario: str
    generated_reply: str
    reference_reply: str
    automated_scores: MetricScores
    judge_scores: Optional[JudgeScore]
    composite_score: float
    retrieved_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "email_id": self.email_id,
            "category": self.category,
            "scenario": self.scenario,
            "generated_reply": self.generated_reply,
            "scores": {
                **self.automated_scores.to_dict(),
                "composite_score": round(self.composite_score, 4),
            },
            "retrieved_examples": self.retrieved_ids,
        }
        if self.judge_scores:
            result["scores"]["llm_judge"] = self.judge_scores.to_dict()
        return result


@dataclass
class EvaluationReport:
    results: List[EvaluationResult]

    @property
    def overall_composite(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.composite_score for r in self.results) / len(self.results)

    def by_category(self) -> Dict[str, float]:
        categories: Dict[str, List[float]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r.composite_score)
        return {cat: sum(scores) / len(scores) for cat, scores in categories.items()}

    def avg_automated_scores(self) -> Dict[str, float]:
        if not self.results:
            return {}
        keys = ["rouge_l", "bert_score", "sentence_similarity", "key_action_coverage"]
        avgs = {}
        for key in keys:
            values = [getattr(r.automated_scores, key) for r in self.results]
            avgs[key] = sum(values) / len(values)
        return avgs

    def avg_judge_scores(self) -> Dict[str, float]:
        judged = [r for r in self.results if r.judge_scores]
        if not judged:
            return {}
        dimensions = ["semantic_accuracy", "intent_alignment", "tone", "completeness", "coherence"]
        avgs = {}
        for dim in dimensions:
            values = [getattr(r.judge_scores, dim)["score"] for r in judged]
            avgs[dim] = sum(values) / len(values)
        return avgs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_composite": round(self.overall_composite, 4),
            "num_evaluated": len(self.results),
            "avg_automated_scores": {k: round(v, 4) for k, v in self.avg_automated_scores().items()},
            "avg_judge_scores": {k: round(v, 2) for k, v in self.avg_judge_scores().items()},
            "by_category": {k: round(v, 4) for k, v in self.by_category().items()},
            "per_response": [r.to_dict() for r in self.results],
        }


class Evaluator:
    def __init__(self, use_llm_judge: bool = True):
        self.metrics = AutomatedMetrics()
        self.judge = LLMJudge() if use_llm_judge else None

    def evaluate_single(
        self,
        email_id: str,
        category: str,
        scenario: str,
        incoming_email: Dict[str, str],
        generated_reply: str,
        reference_reply: str,
        key_actions: List[str] = None,
        retrieved_ids: List[str] = None,
    ) -> EvaluationResult:
        automated = self.metrics.compute_all(generated_reply, reference_reply, key_actions or [])

        judge_scores = None
        if self.judge:
            judge_scores = self.judge.evaluate(
                incoming_from=incoming_email.get("from", ""),
                incoming_subject=incoming_email.get("subject", ""),
                incoming_body=incoming_email.get("body", ""),
                reference_reply=reference_reply,
                generated_reply=generated_reply,
            )

        composite = self._compute_composite(automated, judge_scores)

        return EvaluationResult(
            email_id=email_id,
            category=category,
            scenario=scenario,
            generated_reply=generated_reply,
            reference_reply=reference_reply,
            automated_scores=automated,
            judge_scores=judge_scores,
            composite_score=composite,
            retrieved_ids=retrieved_ids or [],
        )

    def evaluate_batch(self, items: List[Dict[str, Any]]) -> EvaluationReport:
        results = []
        for item in items:
            result = self.evaluate_single(
                email_id=item["email_id"],
                category=item["category"],
                scenario=item.get("scenario", ""),
                incoming_email=item["incoming_email"],
                generated_reply=item["generated_reply"],
                reference_reply=item["reference_reply"],
                key_actions=item.get("key_actions", []),
                retrieved_ids=item.get("retrieved_ids", []),
            )
            results.append(result)
        return EvaluationReport(results=results)

    def _compute_composite(self, automated: MetricScores, judge: Optional[JudgeScore]) -> float:
        score = (
            COMPOSITE_WEIGHTS["rouge_l"] * automated.rouge_l
            + COMPOSITE_WEIGHTS["bert_score"] * automated.bert_score
            + COMPOSITE_WEIGHTS["sentence_similarity"] * automated.sentence_similarity
            + COMPOSITE_WEIGHTS["key_action_coverage"] * automated.key_action_coverage
        )

        if judge:
            score += COMPOSITE_WEIGHTS["llm_judge"] * judge.normalized_average
        else:
            remaining = COMPOSITE_WEIGHTS["llm_judge"]
            auto_total = 1.0 - remaining
            if auto_total > 0:
                score = score / auto_total

        return min(1.0, max(0.0, score))
