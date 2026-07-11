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
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    generation_model: str = "llama-3.3-70b-versatile"
    judge_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "all-MiniLM-L6-v2"
    temperature: float = 0.7
    top_k_retrieval: int = 3
    chroma_collection: str = "email_pairs"
    dataset_path: str = str(DATA_DIR / "email_dataset.json")

    def validate(self):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Please set it in .env or environment variables.")


config = Config()
