import os
from dotenv import load_dotenv

load_dotenv(".env")


DATA_DIR = "_data/"
DOCUMENT_PARQUET_FILE = "02_KRP_selec.parq"

WEAVIATE_INDEX_DIR = "_weaviate_index/"
WEAVIATE_COLLECTION_NAME = "research_app"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_PLATFORM = "mps"  # "cuda" for CUDA GPU, "mps" for Mac, "cpu" for CPU
EMBEDDING_MAX_LENGTH = 500

OPEN_ROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

headroom_ratio = 1.2  # Leave some headroom for prompt and response tokens

MAX_INPUT_TOKENS = {
    "Claude Sonnet 4.6": int(200_000 / headroom_ratio),
    "GPT-5.4": int(128_000 / headroom_ratio),
    "Google Gemini 3.1 Flash": int(400_000 / headroom_ratio),
    "Google Gemini 3.1 Pro": int(400_000 / headroom_ratio),
}

MODEL_CHOICES = {
    "Google Gemini 3.1 Flash": "google/gemini-3-flash-preview",
    "Google Gemini 3.1 Pro": "google/gemini-3.1-pro-preview",
    "Claude Sonnet 4.6": "anthropic/claude-4.6-sonnet",
    "GPT-5.4": "openai/gpt-5.4",
}

MODEL_CHOICES_REVERSE = {v: k for k, v in MODEL_CHOICES.items()}

DEFAULT_MODEL = "Google Gemini 3.1 Flash"

HYBRID_BALANCE = 0.7

INFO_TEXT = """Dies ist ein Test für eine App, mit der du **Dokumente nach Stichworten (*lexikalisch*) und nach Bedeutung (*semantisch*) durchsuchen und mit einem Sprachmodell (LLM) befragen** kannst.\n\nDie App dient zum Testen. **Beachte, dass sowohl die Suche als auch die Antworten fehlerhaft oder unvollständig sein können.** Überprüfe die Ergebnisse immer.\n\nDeine Fragen werden an Clouddienste weitergeleitet und dort verarbeitet. **Gib daher nur als öffentlich klassifizierte Informationen als Fragen bzw. Promptinhalte ein.**.\n\nApp-Version v0.3. Letzte Aktualisierung 15.3.2025"""

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
