import os
from dotenv import load_dotenv

load_dotenv(".env_example")


DATA_DIR = "_data/"
DOCUMENT_PARQUET_FILE = "02_KRP_selec.parq"

WEAVIATE_INDEX_DIR = "_weaviate_index/"
WEAVIATE_COLLECTION_NAME = "research_app"

EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-de"
EMBEDDING_PLATFORM = "mps"  # "cuda" for CUDA GPU, "mps" for Mac, "cpu" for CPU
EMBEDDING_MAX_LENGTH = 1200

OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MAX_INPUT_TOKENS = {
    "Claude Sonnet": 180_000,
    "GPT-4o": 110_000,
    "GPT-4.1-nano": 440_000,
    "GPT-4.1-mini": 220_000,
    "GPT-4.1": 220_000,
    "o4-mini": 110_000,
    "Gemma-3 (OS)": 110_000,
    "Llama 3.3 (OS)": 110_000,
}

MODEL_CHOICES = {
    "Claude Sonnet": "anthropic/claude-3.5-sonnet",
    "GPT-4o": "openai/gpt-4o-2024-11-20",
    "GPT-4.1-nano": "openai/gpt-4.1-nano",
    "GPT-4.1-mini": "openai/gpt-4.1-mini",
    "GPT-4.1": "openai/gpt-4.1",
    "o4-mini": "openai/o4-mini",
    "Gemma-3 (OS)": "google/gemma-3-27b-it",
    "Llama 3.3 (OS)": "meta-llama/llama-3.3-70b-instruct",
}

MODEL_CHOICES_REVERSE = {v: k for k, v in MODEL_CHOICES.items()}

DEFAULT_MODEL = "GPT-4.1-mini"

DEFAULT_TEMPERATURE = 0.2

MODEL_TEMPERATURES = {
    "Claude Sonnet": DEFAULT_TEMPERATURE,
    "GPT-4o": DEFAULT_TEMPERATURE,
    "GPT-4.1-nano": DEFAULT_TEMPERATURE,
    "GPT-4.1-mini": DEFAULT_TEMPERATURE,
    "GPT-4.1": DEFAULT_TEMPERATURE,
    "o4-mini": DEFAULT_TEMPERATURE,
    "Gemma-3 (OS)": 1.0,
    "Llama 3.3 (OS)": DEFAULT_TEMPERATURE,
}


HYBRID_BALANCE = 0.7

INFO_TEXT = """Dies ist ein Test für eine App, mit der du **Dokumente nach Stichworten (*lexikalisch*) und nach Bedeutung (*semantisch*) durchsuchen und mit einem Sprachmodell (LLM) befragen** kannst.\n\nDie App dient zum Testen. **Beachte, dass sowohl die Suche als auch die Antworten fehlerhaft oder unvollständig sein können.** Überprüfe die Ergebnisse immer.\n\nDeine Fragen werden an Clouddienste weitergeleitet und dort verarbeitet. **Gib daher nur als öffentlich klassifizierte Informationen als Fragen bzw. Promptinhalte ein.**.\n\nApp-Version v0.1. Letzte Aktualisierung 23.5.2025"""

INSTRUCTIONS = """#### Tipps zur Bedienung

##### Suche nach Quellen
- Gib im linken Feld **Suchbegriffe oder Fragen** ein. Klicke auf **«Suchen»**. Du erhältst deine Suchergebnisse als Liste unter dem Suchfeld.
- **Wähle einen oder mehrere Quellen aus**, die du an das Sprachmodell schicken willst.
- Du kannst mit der **SHIFT-Taste** mehrere Quellen auswählen.
- Du kannst mit der **CTRL-Taste** (auf Windows) und **CMD-Taste** (auf Mac) mehrere Quellen auswählen, die in der Liste nicht aufeinander folgen.

##### Die ausgewählten Quellen «befragen»
- **Gib im rechten Feld deine Frage oder deinen Prompt ein** und klicke auf **«Fragen»**.
- Du kannst auch Anweisungen geben wie: *„Fasse die Quellen einzeln zusammen“*.
- Du kannst auch **mehrere Anweisungen in einem Prompt geben**, z. B.:
  *„Fasse alle Quellen einzeln zusammen. Liste alle wichtigen Entscheidkriterien auf.“*
- Du kannst **immer wieder neue Fragen oder Anweisungen geben**. Die Antworten beziehen sich weiter auf die ausgewählten Quellen aus dem ersten Schritt.

##### Einstellungen
- Wähle die **Balance zwischen exakter Suche nach Stichwort (lexikalisch) und Suche nach Bedeutung (semantisch)**.
- Die App führt generell beide Suchen aus und fügt die Resultate beider Abfragen in der Trefferliste zusammen.
- Wenn du **ausschließlich exakt nach Stichworten suchen willst**, wähle **0**.
- Wenn du **nur semantische Treffer haben willst**, wähle **1**.
- Beachte, dass eine **semantische Suche immer Ergebnisse liefert**, selbst wenn die Resultate inhaltlich sehr weit von deiner Abfrage entfernt sind.
"""
