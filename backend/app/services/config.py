import os
from pathlib import Path
from dataclasses import dataclass

REPO_ROOT = Path(__file__).resolve().parents[3]


def _pick_default_path(candidates: list[Path]) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


DEFAULT_NER_MODEL_PATH = os.getenv("MED_NER_MODEL_PATH","../Qwen3-4B-Med_ft")
DEFAULT_REASON_MODEL_PATH = os.getenv("MED_REASON_MODEL_PATH","../Qwen3-8B-Instruct")
DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ZYDzyd917917")
DEFAULT_DEVICE = os.getenv("MED_MODEL_DEVICE", "auto")


class PipelineConfigError(RuntimeError):
    pass


@dataclass
class LocalModelConfig:
    model_path: str
    device: str = DEFAULT_DEVICE
    torch_dtype: str = "auto"
