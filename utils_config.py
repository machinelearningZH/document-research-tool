import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(".env")

# Load configuration from config.yaml.
_config_path = Path(__file__).parent / "config.yaml"
with open(_config_path) as f:
    _cfg = yaml.safe_load(f)

# Paths
DATA_DIR = Path(_cfg["paths"]["data_dir"])
DOCUMENT_PARQUET_FILE = _cfg["paths"]["document_parquet_file"]
WEAVIATE_INDEX_DIR = _cfg["paths"]["weaviate_index_dir"]

# Search
WEAVIATE_COLLECTION_NAME = _cfg["search"]["weaviate_collection_name"]
HYBRID_BALANCE = _cfg["search"]["hybrid_balance"]
BM25_LIMIT = _cfg["search"]["bm25_limit"]
HYBRID_LIMIT = _cfg["search"]["hybrid_limit"]

# Embedding
EMBEDDING_MODEL = _cfg["embedding"]["model"]
EMBEDDING_PLATFORM = _cfg["embedding"]["platform"]
EMBEDDING_MAX_LENGTH = _cfg["embedding"]["max_length"]

# LLM
DEFAULT_MODEL = _cfg["llm"]["default_model"]
HEADROOM_RATIO = _cfg["llm"]["headroom_ratio"]
MAX_OUTPUT_TOKENS = _cfg["llm"]["max_output_tokens"]
TIKTOKEN_MODEL = _cfg["llm"]["tiktoken_model"]
MODEL_CHOICES: dict[str, str] = _cfg["llm"]["model_choices"]
MODEL_CHOICES_REVERSE = {v: k for k, v in MODEL_CHOICES.items()}
_context_lengths: dict[str, int] = _cfg["llm"]["context_lengths"]
MAX_INPUT_TOKENS = {k: int(v / HEADROOM_RATIO) for k, v in _context_lengths.items()}

# Validate that all model choices have token limit entries.
_missing = set(MODEL_CHOICES.keys()) - set(MAX_INPUT_TOKENS.keys())
if _missing:
    raise ValueError(f"Missing context_lengths entries for models: {_missing}")

# Secrets
OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPEN_ROUTER_API_KEY:
    logging.warning(
        "OPENROUTER_API_KEY not set in environment. LLM features will not work. "
        "Add it to .env file."
    )

# UI
UI_COLORS: dict[str, str] = _cfg["ui"]["colors"]
INFO_TEXT: str = _cfg["ui"]["info_text"].strip()
INSTRUCTIONS: str = _cfg["ui"]["instructions"].strip()
