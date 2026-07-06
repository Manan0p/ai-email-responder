"""
Generate a synthetic dataset of 100 email-reply pairs for an AI email responder system.

Uses Google Gemini API to expand 20 scenario templates (5 categories x 4 scenarios)
into 5 variations each, yielding 100 realistic email-reply pairs.

Usage:
    python generate_dataset.py
    python generate_dataset.py --output data/email_dataset.json
    python generate_dataset.py --validate
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from google import genai

# ---------------------------------------------------------------------------
# Scenario templates: 5 categories x 4 scenarios each = 20 templates
# ---------------------------------------------------------------------------

SCENARIO_TEMPLATES = {
    "meeting_scheduling": {
        "schedule_request": {
            "description": "Someone requesting to schedule a new meeting",
            "sample_context": "A colleague or external contact wants to set up a meeting to discuss a topic.",
            "tones": ["formal", "casual", "urgent"],
            "complexities": ["low", "medium", "high"],
        },
        "reschedule_request": {
            "description": "Someone asking to reschedule an existing meeting",
            "sample_context": "A previously confirmed meeting needs to be moved due to a conflict.",
            "tones": ["apologetic", "formal", "casual"],
            "complexities": ["low", "medium"],
        },
        "cancel_meeting": {
            "description": "Someone canceling a scheduled meeting",
            "sample_context": "A meeting is no longer needed or cannot happen as planned.",
            "tones": ["formal", "apologetic", "direct"],
            "complexities": ["low", "medium"],
        },
        "confirm_meeting": {
            "description": "Someone confirming attendance or details for a meeting",
            "sample_context": "Confirming time, location, agenda, or attendees for an upcoming meeting.",
            "tones": ["formal", "casual", "enthusiastic"],
            "complexities": ["low", "medium"],
        },
    },
    "project_updates": {
        "status_request": {
            "description": "Manager or stakeholder requesting a project status update",
            "sample_context": "Someone in leadership needs to know where things stand on a project.",
            "tones": ["formal", "direct", "concerned"],
            "complexities": ["medium", "high"],
        },
        "deadline_extension": {
            "description": "Team member requesting an extension on a project deadline",
            "sample_context": "Unforeseen issues require more time to complete deliverables.",
            "tones": ["apologetic", "professional", "urgent"],
            "complexities": ["medium", "high"],
        },
        "blocker_report": {
            "description": "Team member reporting a blocker or impediment",
            "sample_context": "A technical, resource, or dependency issue is preventing progress.",
            "tones": ["urgent", "professional", "frustrated"],
            "complexities": ["medium", "high"],
        },
        "milestone_completion": {
            "description": "Team member announcing completion of a project milestone",
            "sample_context": "A significant phase or deliverable has been completed successfully.",
            "tones": ["enthusiastic", "professional", "formal"],
            "complexities": ["low", "medium"],
        },
    },
    "customer_support": {
        "complaint": {
            "description": "Customer filing a complaint about a product or service",
            "sample_context": "A customer has had a negative experience and is reaching out.",
            "tones": ["frustrated", "angry", "disappointed"],
            "complexities": ["medium", "high"],
        },
        "refund_request": {
            "description": "Customer requesting a refund for a purchase",
            "sample_context": "A customer wants their money back due to dissatisfaction or error.",
            "tones": ["firm", "polite", "frustrated"],
            "complexities": ["medium", "high"],
        },
        "product_inquiry": {
            "description": "Customer asking about product features, availability, or compatibility",
            "sample_context": "A potential or existing customer wants information before a purchase decision.",
            "tones": ["curious", "formal", "casual"],
            "complexities": ["low", "medium"],
        },
        "billing_issue": {
            "description": "Customer reporting a billing discrepancy or payment problem",
            "sample_context": "Unexpected charges, failed payments, or invoice discrepancies.",
            "tones": ["concerned", "frustrated", "formal"],
            "complexities": ["medium", "high"],
        },
    },
    "hr_internal": {
        "pto_request": {
            "description": "Employee requesting paid time off",
            "sample_context": "An employee needs to take vacation, sick leave, or personal days.",
            "tones": ["casual", "formal", "apologetic"],
            "complexities": ["low", "medium"],
        },
        "onboarding_question": {
            "description": "New hire asking about onboarding processes or resources",
            "sample_context": "A recently hired employee needs help navigating their first weeks.",
            "tones": ["curious", "eager", "formal"],
            "complexities": ["low", "medium"],
        },
        "policy_clarification": {
            "description": "Employee asking for clarification on a company policy",
            "sample_context": "An employee needs to understand remote work, expense, or conduct policies.",
            "tones": ["formal", "casual", "concerned"],
            "complexities": ["medium", "high"],
        },
        "feedback": {
            "description": "Employee providing or requesting feedback on performance or processes",
            "sample_context": "Peer review, manager feedback, or process improvement suggestions.",
            "tones": ["constructive", "professional", "diplomatic"],
            "complexities": ["medium", "high"],
        },
    },
    "sales_partnership": {
        "cold_outreach_response": {
            "description": "Responding to a cold sales or partnership outreach email",
            "sample_context": "Someone received an unsolicited business proposal and needs to reply.",
            "tones": ["polite", "interested", "noncommittal"],
            "complexities": ["low", "medium"],
        },
        "proposal_followup": {
            "description": "Following up on a previously sent business proposal",
            "sample_context": "Checking in after sending a proposal to gauge interest or next steps.",
            "tones": ["professional", "persistent", "friendly"],
            "complexities": ["medium", "high"],
        },
        "pricing_negotiation": {
            "description": "Negotiating pricing or terms for a deal",
            "sample_context": "Back-and-forth on pricing, discounts, or contract terms.",
            "tones": ["firm", "diplomatic", "persuasive"],
            "complexities": ["high"],
        },
        "contract_question": {
            "description": "Asking about contract terms, renewals, or amendments",
            "sample_context": "Clarifying legal language, renewal dates, or modification requests.",
            "tones": ["formal", "cautious", "direct"],
            "complexities": ["medium", "high"],
        },
    },
}

EXPECTED_CATEGORIES = set(SCENARIO_TEMPLATES.keys())

# ---------------------------------------------------------------------------
# Name / company pools for realistic variation
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "James", "Maria", "David", "Sarah", "Michael", "Priya", "Carlos",
    "Emily", "Chen", "Fatima", "Robert", "Lisa", "Ahmed", "Jennifer",
    "Kenji", "Olivia", "Andre", "Rachel", "Yuki", "Thomas",
]

LAST_NAMES = [
    "Johnson", "Patel", "Williams", "Garcia", "Chen", "Smith", "Kim",
    "Mueller", "Santos", "Nakamura", "Brown", "Singh", "Taylor", "Ali",
    "Anderson", "Martinez", "Lee", "Wilson", "Thompson", "Clark",
]

COMPANIES = [
    "Apex Dynamics", "Brightwave Solutions", "Cascadia Tech", "DataForge Inc.",
    "Evergreen Systems", "Frontier Analytics", "GlobalSync Corp", "HorizonX",
    "InnoVault", "Keystone Partners", "Luminar AI", "Meridian Group",
    "NexGen Software", "Orbit Labs", "Pinnacle Consulting", "Quantis Research",
    "Redwood Digital", "Stellar Platforms", "TrueNorth Ventures", "Uplift Media",
]


def _random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _random_email(name, company):
    first, last = name.lower().split(" ", 1)
    domain = company.lower().replace(" ", "").replace(".", "").replace(",", "")
    return f"{first}.{last}@{domain}.com"


# ---------------------------------------------------------------------------
# Gemini generation
# ---------------------------------------------------------------------------


def build_prompt(category: str, scenario: str, template: dict, variation_index: int) -> str:
    """Build a prompt for Gemini to generate one email-reply pair."""
    sender_name = _random_name()
    sender_company = random.choice(COMPANIES)
    recipient_name = _random_name()
    recipient_company = random.choice(COMPANIES)

    tone = random.choice(template["tones"])
    complexity = random.choice(template["complexities"])

    return f"""Generate a single realistic email and its professional reply for an AI email responder training dataset.

SCENARIO DETAILS:
- Category: {category}
- Scenario type: {scenario}
- Description: {template["description"]}
- Context: {template["sample_context"]}
- Variation number: {variation_index + 1} of 5 (make this variation distinct from others)

SENDER: {sender_name} from {sender_company} ({_random_email(sender_name, sender_company)})
RECIPIENT: {recipient_name} from {recipient_company} ({_random_email(recipient_name, recipient_company)})

REQUIREMENTS:
- Tone of the incoming email: {tone}
- Complexity level: {complexity}
- The reply should be helpful, professional, and actionable
- Include specific details (dates, project names, product names, dollar amounts where relevant)
- The email body should be 3-8 sentences long
- The reply body should be 3-10 sentences long
- Make it feel like a real workplace email, not a template

Return ONLY a valid JSON object with this exact structure (no markdown, no code fences):
{{
  "incoming_email": {{
    "from": "{sender_name} <{_random_email(sender_name, sender_company)}>",
    "to": "{recipient_name} <{_random_email(recipient_name, recipient_company)}>",
    "subject": "a realistic email subject line",
    "body": "the full email body text"
  }},
  "expected_reply": {{
    "subject": "Re: matching the incoming subject",
    "body": "the full reply body text"
  }},
  "metadata": {{
    "tone": "{tone}",
    "intent": "a 2-5 word description of the core intent",
    "key_actions": ["action1", "action2"],
    "complexity": "{complexity}"
  }}
}}"""


def generate_variation(
    client: genai.Client,
    category: str,
    scenario: str,
    template: dict,
    variation_index: int,
    entry_id: str,
    split: str,
    max_retries: int = 3,
) -> dict:
    """Call Gemini to generate one email-reply pair and wrap it in the full schema."""
    prompt = build_prompt(category, scenario, template, variation_index)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.0-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.8,
                    response_mime_type="application/json",
                ),
            )

            raw_text = response.text.strip()
            # Remove potential markdown fences if the model wraps them anyway
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]
            if raw_text.endswith("```"):
                raw_text = raw_text.rsplit("```", 1)[0]
            raw_text = raw_text.strip()

            generated = json.loads(raw_text)

            entry = {
                "id": entry_id,
                "category": category,
                "scenario": scenario,
                "split": split,
                "incoming_email": generated["incoming_email"],
                "expected_reply": generated["expected_reply"],
                "metadata": generated["metadata"],
            }
            return entry

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"  [!] Attempt {attempt + 1}/{max_retries} failed for {entry_id}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"  [ERROR] All retries exhausted for {entry_id}. Using fallback.")
                return _fallback_entry(entry_id, category, scenario, split, template)
        except Exception as exc:
            print(f"  [!] API error on attempt {attempt + 1}/{max_retries} for {entry_id}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  [ERROR] All retries exhausted for {entry_id}. Using fallback.")
                return _fallback_entry(entry_id, category, scenario, split, template)


def _fallback_entry(entry_id: str, category: str, scenario: str, split: str, template: dict) -> dict:
    """Return a minimal valid entry when Gemini generation fails."""
    sender_name = _random_name()
    sender_company = random.choice(COMPANIES)
    recipient_name = _random_name()
    recipient_company = random.choice(COMPANIES)
    tone = random.choice(template["tones"])
    complexity = random.choice(template["complexities"])

    subject = f"{template['description']}"
    return {
        "id": entry_id,
        "category": category,
        "scenario": scenario,
        "split": split,
        "incoming_email": {
            "from": f"{sender_name} <{_random_email(sender_name, sender_company)}>",
            "to": f"{recipient_name} <{_random_email(recipient_name, recipient_company)}>",
            "subject": subject,
            "body": f"Dear {recipient_name},\n\n{template['sample_context']}\n\nPlease advise on the next steps.\n\nBest regards,\n{sender_name}",
        },
        "expected_reply": {
            "subject": f"Re: {subject}",
            "body": f"Dear {sender_name},\n\nThank you for reaching out. I have noted your request regarding {template['description'].lower()} and will follow up shortly with the relevant details.\n\nBest regards,\n{recipient_name}",
        },
        "metadata": {
            "tone": tone,
            "intent": scenario.replace("_", " "),
            "key_actions": ["acknowledge", "follow up"],
            "complexity": complexity,
        },
    }


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def assign_splits() -> dict:
    """
    Decide which (category, scenario, variation_index) entries are test vs train.
    4 test entries per category = 20 test total, 80 train total.
    Within each category (20 entries), pick 4 at random for test.
    """
    splits = {}
    for category in SCENARIO_TEMPLATES:
        # Each category has 4 scenarios x 5 variations = 20 entries
        indices = []
        for scenario in SCENARIO_TEMPLATES[category]:
            for v in range(5):
                indices.append((category, scenario, v))
        test_indices = set(random.sample(range(len(indices)), 4))
        for i, key in enumerate(indices):
            splits[key] = "test" if i in test_indices else "train"
    return splits


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_TOP_FIELDS = {"id", "category", "scenario", "split", "incoming_email", "expected_reply", "metadata"}
REQUIRED_EMAIL_FIELDS = {"from", "to", "subject", "body"}
REQUIRED_REPLY_FIELDS = {"subject", "body"}
REQUIRED_METADATA_FIELDS = {"tone", "intent", "key_actions", "complexity"}


def validate_dataset(dataset: list[dict]) -> list[str]:
    """
    Validate the dataset against the expected schema.

    Returns a list of error strings. An empty list means the dataset is valid.
    """
    errors = []

    if not isinstance(dataset, list):
        errors.append("Dataset must be a list of entries.")
        return errors

    if len(dataset) != 100:
        errors.append(f"Expected 100 entries, got {len(dataset)}.")

    # Check unique IDs
    ids = [e.get("id") for e in dataset]
    if len(set(ids)) != len(ids):
        duplicates = [x for x in ids if ids.count(x) > 1]
        errors.append(f"Duplicate IDs found: {set(duplicates)}")

    # Check split distribution
    train_count = sum(1 for e in dataset if e.get("split") == "train")
    test_count = sum(1 for e in dataset if e.get("split") == "test")
    if train_count != 80:
        errors.append(f"Expected 80 train entries, got {train_count}.")
    if test_count != 20:
        errors.append(f"Expected 20 test entries, got {test_count}.")

    # Check per-category test distribution
    category_test_counts = {}
    for e in dataset:
        cat = e.get("category", "UNKNOWN")
        if e.get("split") == "test":
            category_test_counts[cat] = category_test_counts.get(cat, 0) + 1
    for cat in EXPECTED_CATEGORIES:
        count = category_test_counts.get(cat, 0)
        if count != 4:
            errors.append(f"Category '{cat}' has {count} test entries (expected 4).")

    # Check each entry
    for i, entry in enumerate(dataset):
        prefix = f"Entry {i} (id={entry.get('id', 'MISSING')})"

        # Top-level fields
        missing_top = REQUIRED_TOP_FIELDS - set(entry.keys())
        if missing_top:
            errors.append(f"{prefix}: Missing top-level fields: {missing_top}")
            continue

        # Category check
        if entry["category"] not in EXPECTED_CATEGORIES:
            errors.append(f"{prefix}: Unknown category '{entry['category']}'.")

        # Split check
        if entry["split"] not in ("train", "test"):
            errors.append(f"{prefix}: Invalid split '{entry['split']}'.")

        # Incoming email fields
        if isinstance(entry.get("incoming_email"), dict):
            missing_email = REQUIRED_EMAIL_FIELDS - set(entry["incoming_email"].keys())
            if missing_email:
                errors.append(f"{prefix}: incoming_email missing fields: {missing_email}")
        else:
            errors.append(f"{prefix}: incoming_email is not a dict.")

        # Expected reply fields
        if isinstance(entry.get("expected_reply"), dict):
            missing_reply = REQUIRED_REPLY_FIELDS - set(entry["expected_reply"].keys())
            if missing_reply:
                errors.append(f"{prefix}: expected_reply missing fields: {missing_reply}")
        else:
            errors.append(f"{prefix}: expected_reply is not a dict.")

        # Metadata fields
        if isinstance(entry.get("metadata"), dict):
            missing_meta = REQUIRED_METADATA_FIELDS - set(entry["metadata"].keys())
            if missing_meta:
                errors.append(f"{prefix}: metadata missing fields: {missing_meta}")
            else:
                if not isinstance(entry["metadata"].get("key_actions"), list):
                    errors.append(f"{prefix}: metadata.key_actions must be a list.")
                if entry["metadata"].get("complexity") not in ("low", "medium", "high"):
                    errors.append(f"{prefix}: metadata.complexity must be low|medium|high.")
        else:
            errors.append(f"{prefix}: metadata is not a dict.")

    return errors


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------


def generate_dataset(api_key: str) -> list[dict]:
    """Generate the full 100-entry dataset using Gemini."""
    client = genai.Client(api_key=api_key)

    splits = assign_splits()
    dataset = []
    entry_number = 0

    total = sum(
        len(scenarios) * 5 for scenarios in SCENARIO_TEMPLATES.values()
    )

    print(f"Generating {total} email-reply pairs across {len(SCENARIO_TEMPLATES)} categories...\n")

    for category, scenarios in SCENARIO_TEMPLATES.items():
        print(f"[{category.upper()}]")
        for scenario, template in scenarios.items():
            for variation in range(5):
                entry_number += 1
                entry_id = f"email_{entry_number:03d}"
                split = splits[(category, scenario, variation)]

                print(f"  ({entry_number}/{total}) {entry_id} | {scenario} v{variation + 1} [{split}] ... ", end="", flush=True)

                entry = generate_variation(
                    client=client,
                    category=category,
                    scenario=scenario,
                    template=template,
                    variation_index=variation,
                    entry_id=entry_id,
                    split=split,
                )
                dataset.append(entry)
                print("done")

                # Brief pause to avoid rate-limiting
                time.sleep(0.3)

        print()

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic email-reply dataset using Google Gemini."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/email_dataset.json",
        help="Output file path (default: data/email_dataset.json)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run schema validation on an existing dataset file instead of generating.",
    )
    args = parser.parse_args()

    # --validate mode: load and check an existing file
    if args.validate:
        target = args.output
        if not os.path.exists(target):
            print(f"File not found: {target}")
            sys.exit(1)
        with open(target, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Validating {target} ({len(dataset)} entries)...\n")
        errors = validate_dataset(dataset)
        if errors:
            print(f"Validation FAILED with {len(errors)} error(s):\n")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print("Validation PASSED. Dataset conforms to the expected schema.")
            # Print summary stats
            cats = {}
            for e in dataset:
                cats.setdefault(e["category"], {"train": 0, "test": 0})
                cats[e["category"]][e["split"]] += 1
            print(f"\n{'Category':<25} {'Train':>6} {'Test':>6} {'Total':>6}")
            print("-" * 45)
            for cat in sorted(cats):
                t, te = cats[cat]["train"], cats[cat]["test"]
                print(f"{cat:<25} {t:>6} {te:>6} {t + te:>6}")
            total_train = sum(c["train"] for c in cats.values())
            total_test = sum(c["test"] for c in cats.values())
            print("-" * 45)
            print(f"{'TOTAL':<25} {total_train:>6} {total_test:>6} {total_train + total_test:>6}")
        sys.exit(0)

    # Load API key
    if load_dotenv is not None:
        load_dotenv()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found.")
        print("Set it as an environment variable or add it to a .env file.")
        sys.exit(1)

    # Generate
    dataset = generate_dataset(api_key)

    # Validate before saving
    print("Running validation on generated dataset...")
    errors = validate_dataset(dataset)
    if errors:
        print(f"\nWARNING: Generated dataset has {len(errors)} validation issue(s):")
        for err in errors:
            print(f"  - {err}")
        print("\nSaving anyway -- review and re-run if needed.\n")
    else:
        print("Validation passed.\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to {output_path.resolve()}")
    print(f"  Total entries: {len(dataset)}")
    print(f"  Train: {sum(1 for e in dataset if e['split'] == 'train')}")
    print(f"  Test:  {sum(1 for e in dataset if e['split'] == 'test')}")


if __name__ == "__main__":
    main()
