"""Tests for the evaluation system, including controlled degradation tests."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import AutomatedMetrics
from src.config import config


@pytest.fixture
def metrics():
    return AutomatedMetrics()


class TestRougeL:
    def test_identical_texts(self, metrics):
        score = metrics.rouge_l("Hello world", "Hello world")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_completely_different(self, metrics):
        score = metrics.rouge_l("The cat sat on the mat", "Purple elephants dance freely")
        assert score < 0.3

    def test_partial_overlap(self, metrics):
        score = metrics.rouge_l(
            "Thank you for reaching out about the refund",
            "Thank you for contacting us regarding the refund request"
        )
        assert 0.3 < score < 1.0


class TestBERTScore:
    def test_identical_texts(self, metrics):
        score = metrics.compute_bert_score("Hello world", "Hello world")
        assert score > 0.8

    def test_semantic_similarity(self, metrics):
        score = metrics.compute_bert_score(
            "I'm sorry to hear about the issue with your order",
            "I apologize for the problems you experienced with your purchase"
        )
        assert score > 0.3

    def test_unrelated_texts(self, metrics):
        score_related = metrics.compute_bert_score(
            "Let's schedule a meeting for tomorrow",
            "Can we set up a call for tomorrow?"
        )
        score_unrelated = metrics.compute_bert_score(
            "Let's schedule a meeting for tomorrow",
            "The annual rainfall in the Amazon is very high"
        )
        assert score_related > score_unrelated


class TestSentenceSimilarity:
    def test_identical_texts(self, metrics):
        score = metrics.sentence_similarity("Hello world", "Hello world")
        assert score > 0.99

    def test_similar_texts(self, metrics):
        score = metrics.sentence_similarity(
            "I'd like to schedule a meeting for next Tuesday",
            "Can we set up a meeting next Tuesday?"
        )
        assert score > 0.7

    def test_dissimilar_texts(self, metrics):
        score = metrics.sentence_similarity(
            "Please process my refund for the damaged item",
            "The quarterly earnings report shows strong growth"
        )
        assert score < 0.5


class TestKeyActionCoverage:
    def test_all_actions_covered(self, metrics):
        reply = "I sincerely apologize for the inconvenience. We will process your refund within 3-5 business days."
        actions = ["apologize", "offer_refund_or_replacement", "provide_timeline"]
        score = metrics.key_action_coverage(reply, actions)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_no_actions_covered(self, metrics):
        reply = "Thank you for your email. We have noted your feedback."
        actions = ["offer_refund_or_replacement", "provide_timeline", "escalate"]
        score = metrics.key_action_coverage(reply, actions)
        assert score < 0.5

    def test_partial_coverage(self, metrics):
        reply = "I apologize for the issue. Our team is looking into it."
        actions = ["apologize", "offer_refund_or_replacement", "provide_timeline"]
        score = metrics.key_action_coverage(reply, actions)
        assert 0.2 < score < 0.8

    def test_empty_actions(self, metrics):
        score = metrics.key_action_coverage("Any reply text", [])
        assert score == 1.0


class TestCompositeMetrics:
    def test_compute_all(self, metrics):
        result = metrics.compute_all(
            "Thank you for your email. I will schedule the meeting.",
            "Thanks for reaching out. Let me arrange the meeting.",
            ["express_gratitude", "confirm_meeting"]
        )
        assert 0 <= result.rouge_l <= 1
        assert 0 <= result.bert_score <= 1
        assert 0 <= result.sentence_similarity <= 1
        assert 0 <= result.key_action_coverage <= 1

    def test_to_dict(self, metrics):
        result = metrics.compute_all("Hello", "Hello", [])
        d = result.to_dict()
        assert "rouge_l" in d
        assert "bert_score" in d
        assert "sentence_similarity" in d
        assert "key_action_coverage" in d


class TestControlledDegradation:
    """Meta-evaluation: verify that metrics respond correctly to quality changes."""

    def test_perfect_baseline(self, metrics):
        """Scoring reference against itself should yield high scores."""
        text = "Dear Jane, I sincerely apologize for the damaged product. We will process a full refund within 3-5 business days."
        result = metrics.compute_all(text, text, ["apologize", "offer_refund_or_replacement", "provide_timeline"])
        assert result.rouge_l > 0.95
        assert result.sentence_similarity > 0.99
        assert result.key_action_coverage == 1.0

    def test_paraphrase_scores_high(self, metrics):
        """Paraphrased reply should still score high on semantic metrics."""
        reference = "Dear Jane, I sincerely apologize for the damaged product. We will process a full refund within 3-5 business days."
        paraphrase = "Hi Jane, I'm truly sorry about the damage to your item. A complete refund will be issued and should arrive in 3 to 5 working days."
        result = metrics.compute_all(paraphrase, reference, ["apologize", "offer_refund_or_replacement", "provide_timeline"])
        assert result.sentence_similarity > 0.6
        assert result.key_action_coverage >= 0.66

    def test_wrong_intent_scores_low(self, metrics):
        """Replying with wrong intent should score low."""
        reference = "Dear Jane, I sincerely apologize for the damaged product. We will process a full refund within 3-5 business days."
        wrong_intent = "Hi Jane, I'd like to schedule a meeting for next Tuesday at 2 PM. Please confirm your availability."
        result = metrics.compute_all(wrong_intent, reference, ["apologize", "offer_refund_or_replacement", "provide_timeline"])
        assert result.sentence_similarity < 0.5
        assert result.key_action_coverage < 0.5

    def test_incomplete_reply_scores_low_coverage(self, metrics):
        """Reply missing key actions should score low on coverage."""
        reference = "Dear Jane, I sincerely apologize for the damaged product. We will process a full refund within 3-5 business days."
        incomplete = "Dear Jane, thank you for contacting us."
        actions = ["apologize", "offer_refund_or_replacement", "provide_timeline"]
        result = metrics.compute_all(incomplete, reference, actions)
        assert result.key_action_coverage < 0.5


class TestLLMJudge:
    @pytest.mark.skipif(not config.google_api_key, reason="GOOGLE_API_KEY not set")
    def test_judge_returns_scores(self):
        from evaluation.llm_judge import LLMJudge

        judge = LLMJudge()
        result = judge.evaluate(
            incoming_from="customer@example.com",
            incoming_subject="Refund Request",
            incoming_body="I want a refund for my damaged order.",
            reference_reply="I apologize for the damage. We will process your refund within 3-5 business days.",
            generated_reply="I'm sorry about the issue. A full refund will be processed shortly.",
        )

        assert 1 <= result.semantic_accuracy["score"] <= 5
        assert 1 <= result.intent_alignment["score"] <= 5
        assert 1 <= result.tone["score"] <= 5
        assert 1 <= result.completeness["score"] <= 5
        assert 1 <= result.coherence["score"] <= 5
        assert "reason" in result.semantic_accuracy

    @pytest.mark.skipif(not config.google_api_key, reason="GOOGLE_API_KEY not set")
    def test_judge_distinguishes_quality(self):
        from evaluation.llm_judge import LLMJudge

        judge = LLMJudge()

        good = judge.evaluate(
            incoming_from="customer@example.com",
            incoming_subject="Refund Request",
            incoming_body="I want a refund for my damaged order #12345.",
            reference_reply="I apologize for the damage. We will process your refund for order #12345 within 3-5 business days.",
            generated_reply="I'm sorry about the damaged item in order #12345. A full refund will be processed within 3-5 business days.",
        )

        bad = judge.evaluate(
            incoming_from="customer@example.com",
            incoming_subject="Refund Request",
            incoming_body="I want a refund for my damaged order #12345.",
            reference_reply="I apologize for the damage. We will process your refund for order #12345 within 3-5 business days.",
            generated_reply="Hey! Let's schedule a team meeting for Friday. Bring your laptops!",
        )

        assert good.average > bad.average
