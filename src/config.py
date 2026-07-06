import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / ".chroma_db"


@dataclass
class Config:
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    generation_model: str = "gemini-2.0-flash"
    judge_model: str = "gemini-2.0-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    temperature: float = 0.7
    top_k_retrieval: int = 3
    chroma_collection: str = "email_pairs"
    dataset_path: str = str(DATA_DIR / "email_dataset.json")

    def validate(self):
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Get one from https://aistudio.google.com/ "
                "and add it to your .env file."
            )


config = Config()
