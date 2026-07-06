"""Tests for dataset validation."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR

DATASET_PATH = DATA_DIR / "email_dataset.json"

EXPECTED_CATEGORIES = {
    "meeting_scheduling",
    "project_updates",
    "customer_support",
    "hr_internal",
    "sales_partnership",
}

REQUIRED_FIELDS = ["id", "category", "scenario", "incoming_email", "expected_reply", "metadata", "split"]
REQUIRED_EMAIL_FIELDS = ["from", "to", "subject", "body"]
REQUIRED_REPLY_FIELDS = ["subject", "body"]
REQUIRED_METADATA_FIELDS = ["tone", "intent", "key_actions", "complexity"]


@pytest.fixture
def dataset():
    assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}. Run generate_dataset.py first."
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_dataset_not_empty(dataset):
    assert len(dataset) > 0, "Dataset is empty"


def test_dataset_size(dataset):
    assert len(dataset) == 100, f"Expected 100 entries, got {len(dataset)}"


def test_unique_ids(dataset):
    ids = [d["id"] for d in dataset]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {len(ids)} total, {len(set(ids))} unique"


def test_required_fields(dataset):
    for item in dataset:
        for field in REQUIRED_FIELDS:
            assert field in item, f"Missing field '{field}' in {item.get('id', 'unknown')}"


def test_email_fields(dataset):
    for item in dataset:
        email = item["incoming_email"]
        for field in REQUIRED_EMAIL_FIELDS:
            assert field in email, f"Missing email field '{field}' in {item['id']}"
            assert len(email[field]) > 0, f"Empty email field '{field}' in {item['id']}"


def test_reply_fields(dataset):
    for item in dataset:
        reply = item["expected_reply"]
        for field in REQUIRED_REPLY_FIELDS:
            assert field in reply, f"Missing reply field '{field}' in {item['id']}"
            assert len(reply[field]) > 0, f"Empty reply field '{field}' in {item['id']}"


def test_metadata_fields(dataset):
    for item in dataset:
        meta = item["metadata"]
        for field in REQUIRED_METADATA_FIELDS:
            assert field in meta, f"Missing metadata field '{field}' in {item['id']}"


def test_valid_categories(dataset):
    for item in dataset:
        assert item["category"] in EXPECTED_CATEGORIES, (
            f"Invalid category '{item['category']}' in {item['id']}"
        )


def test_all_categories_present(dataset):
    categories = set(d["category"] for d in dataset)
    assert categories == EXPECTED_CATEGORIES, (
        f"Missing categories: {EXPECTED_CATEGORIES - categories}"
    )


def test_split_distribution(dataset):
    train = [d for d in dataset if d["split"] == "train"]
    test = [d for d in dataset if d["split"] == "test"]
    assert len(train) == 80, f"Expected 80 train, got {len(train)}"
    assert len(test) == 20, f"Expected 20 test, got {len(test)}"


def test_key_actions_is_list(dataset):
    for item in dataset:
        actions = item["metadata"]["key_actions"]
        assert isinstance(actions, list), (
            f"key_actions should be list, got {type(actions)} in {item['id']}"
        )
        assert len(actions) > 0, f"key_actions is empty in {item['id']}"


def test_valid_complexity(dataset):
    valid = {"low", "medium", "high"}
    for item in dataset:
        assert item["metadata"]["complexity"] in valid, (
            f"Invalid complexity '{item['metadata']['complexity']}' in {item['id']}"
        )


def test_category_distribution(dataset):
    from collections import Counter
    counts = Counter(d["category"] for d in dataset)
    for cat in EXPECTED_CATEGORIES:
        assert counts[cat] == 20, f"Category '{cat}' has {counts[cat]} entries, expected 20"
