import logging
import os
import time
from datetime import datetime

import pandas as pd
import tiktoken
import weaviate
import weaviate.classes as wvc
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from shiny import App, reactive, render, ui

from utils_config import (
    BM25_LIMIT,
    DATA_DIR,
    DEFAULT_MODEL,
    DOCUMENT_PARQUET_FILE,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL,
    EMBEDDING_PLATFORM,
    HYBRID_BALANCE,
    HYBRID_LIMIT,
    INFO_TEXT,
    INSTRUCTIONS,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MODEL_CHOICES,
    MODEL_CHOICES_REVERSE,
    OPEN_ROUTER_API_KEY,
    TIKTOKEN_MODEL,
    UI_COLORS,
    WEAVIATE_COLLECTION_NAME,
    WEAVIATE_INDEX_DIR,
)
from utils_prompts import BASE_PROMPT

# Suppress Hugging Face warning about tokenizers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(
    level=logging.INFO,
    datefmt="%d-%b-%y %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)


# ---------------------------------------------------------------
# Init

# Load the documents that we will submit to the LLM.
df = pd.read_parquet(DATA_DIR / DOCUMENT_PARQUET_FILE)

openai_async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=OPEN_ROUTER_API_KEY
)


# ---------------------------------------------------------------
# Functions


def log_interaction(
    selected_rows_index: list,
    query: str,
    answer: str,
    model_choice: str,
    success: bool,
    start_time: float,
) -> None:
    """Log interaction."""
    elapsed = time.time() - start_time
    logging.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{selected_rows_index}\t{query}\t{answer}\t{model_choice}\t{success}\t{elapsed:.3f}"
    )


def get_embedding_model():
    """Load the embedding model."""
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        trust_remote_code=True,
        device=EMBEDDING_PLATFORM,  # Use "cuda" for CUDA GPU, "mps" for Mac, "cpu" for CPU.
    )
    model.max_seq_length = EMBEDDING_MAX_LENGTH
    return model


embedding_model = get_embedding_model()


def embed_documents(text):
    """Embed text using the embedding model."""
    try:
        return embedding_model.encode(
            text,
            batch_size=1,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=EMBEDDING_PLATFORM,
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        return None


_tiktoken_encoding = tiktoken.encoding_for_model(TIKTOKEN_MODEL)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    return len(_tiktoken_encoding.encode(string))


try:
    client = weaviate.connect_to_embedded(persistence_data_path=WEAVIATE_INDEX_DIR)
    logging.info("Connected to Weaviate embedded...")
except Exception as e:
    logging.error(f"Error: {e}")
    client = weaviate.connect_to_local(
        port=8079,
        grpc_port=50050,
    )
    logging.info("Connected to Weaviate local...")

collection = client.collections.get(WEAVIATE_COLLECTION_NAME)


def retrieve_ranked_chunks(
    query: str,
    hybrid_balance: reactive.Value,
) -> tuple[list[str], list[str], int]:
    """Retrieve relevant chunks from the data."""
    if not query or not query.strip():
        return [], [], 0

    try:
        embedding = embed_documents(query)
        if embedding is None:
            logging.error("Failed to generate embedding for query")
            return [], [], 0

        alpha = hybrid_balance.get()

        # Only run separate BM25 search for lexical count when not pure semantic.
        response_bm25_count = None
        if alpha < 1.0:
            response_bm25 = collection.query.bm25(
                query=query,
                limit=BM25_LIMIT,
            )
            response_bm25_count = len(response_bm25.objects) if response_bm25.objects else 0

        # Perform the actual hybrid search.
        response = collection.query.hybrid(
            query=query,
            query_properties=["text"],
            vector=embedding,
            limit=HYBRID_LIMIT,
            alpha=alpha,
            fusion_type=wvc.query.HybridFusion.RELATIVE_SCORE,
        )

        result_index = []
        result_chunks = []
        if response.objects is not None:
            for result in response.objects:
                if result.properties["identifier"] in result_index:
                    continue
                result_index.append(result.properties["identifier"])
                result_chunks.append(result.properties["text"])
        return (
            result_index,
            result_chunks,
            response_bm25_count,
        )
    except Exception as e:
        logging.error(f"Error in retrieve_ranked_chunks: {str(e)}")
        return [], [], 0


async def call_openai(
    prompt: str,
    model_id: str = MODEL_CHOICES[DEFAULT_MODEL],
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> str | None:
    """Call the OpenRouter API with appropriate parameters based on the model."""
    if not prompt or not prompt.strip():
        logging.warning("Empty prompt provided to call_openai")
        return None

    try:
        params = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Only add max_tokens for models that support it.
        if model_id != "openai/o4-mini":
            params["max_tokens"] = max_tokens

        completion = await openai_async_client.chat.completions.create(**params)
        return completion.choices[0].message.content

    except Exception as e:
        logging.error(f"Error in call_openai: {str(e)}")
        return None


async def chat_with_decisions(
    query: str, selected_rows: pd.DataFrame, model_choice: str
) -> str:
    """Process user query with selected document rows."""
    if not query or selected_rows is None or selected_rows.empty:
        return "<p>Bitte gib eine Frage ein und wähle mindestens ein Dokument aus.</p>"

    try:
        model_name = MODEL_CHOICES_REVERSE.get(model_choice)
        if not model_name:
            return f"<p>Unbekanntes Modell: {model_choice}</p>"

        # Create context from selected documents.
        context = "".join(
            f"Quelle: {row.title}\n{row.text}\n\n####################\n\n"
            for _, row in selected_rows.iterrows()
        )

        # Check token count of composed context.
        num_tokens = num_tokens_from_string(context)
        max_tokens = MAX_INPUT_TOKENS.get(model_name, 7_800)
        if num_tokens > max_tokens:
            limits = ", ".join(f"{v:,.0f} für {k}" for k, v in MAX_INPUT_TOKENS.items())
            return (
                f"<p>Die ausgewählten Dokumente enthalten insgesamt <strong>{num_tokens:,.0f} Tokens</strong> "
                f"und damit <strong>zu viel Text für die Abfrage</strong>. Bitte wähle weniger Dokumente aus.</p>"
                f"<p>Die Limite betragen momentan: {limits}.</p>"
                f"<p>Bitte beachte, dass <strong>zuviele Inhalte die Antwortqualität verschlechtern</strong> "
                f"und nicht verbessern. Es ist essentiell, möglichst wenige, treffende, relevante Inhalte "
                f"auszuwählen und kein unnötiges «Informationsrauschen» an die Modelle zu schicken.</p>"
            )

        # Get answer from model.
        start_time = time.time()
        prompt = BASE_PROMPT.format(context=context, question=query)
        answer = await call_openai(prompt, model_id=model_choice)

        if answer is None:
            log_interaction(
                selected_rows.index.tolist(),
                query,
                "ERROR: Keine Antwort erhalten",
                model_choice,
                False,
                start_time,
            )
            return "Die Abfrage hat leider nicht funktioniert. Versuche es bitte erneut."

        # Clean up and format answer.
        answer = answer.replace("```html", "").replace("```", "").strip()

        log_interaction(
            selected_rows.index.tolist(),
            query,
            answer,
            model_choice,
            True,
            start_time,
        )
        return answer

    except Exception as e:
        logging.error(f"Error in chat_with_decisions: {str(e)}")
        return f"Ein Fehler ist aufgetreten: {str(e)}"


# ---------------------------------------------------------------
# UI

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.markdown("Recherchetool Kanton Zürich"),
        ui.input_action_button(
            "show_instructions", "👋 Tipps zur Bedienung", class_="btn-sm btn-info"
        ),
        ui.input_slider(
            "hybrid_balance",
            "Balance lexikalisch/semantisch",
            min=0,
            max=1,
            value=HYBRID_BALANCE,
            step=0.1,
        ),
        ui.input_select(
            "model_choice",
            "Sprachmodell",
            {v: k for k, v in MODEL_CHOICES.items()},
            selected=DEFAULT_MODEL,
        ),
        ui.input_action_button("show_appinfo", "Infos zur App", class_="btn-sm"),
        style=f"background:{UI_COLORS['sidebar']} !important;",
    ),
    ui.layout_columns(
        # Left column for Search.
        ui.card(
            ui.div(
                ui.input_text_area(
                    "search_query",
                    "Gib hier Suchbegriffe ein:",
                    value="Was waren wichtige Entscheide zu kantonalen Steuern?",
                    width="100%",
                    rows=2,
                ),
                ui.layout_columns(
                    ui.input_action_button(
                        "search_btn", "Suchen", width="100%", class_="btn-sm btn-warning"
                    ),
                    ui.input_action_button(
                        "copy_btn",
                        "Suchbegriffe zu Prompt kopieren >>",
                        width="100%",
                        class_="btn-sm btn-outline-secondary",
                    ),
                    col_widths=[4, -4, 4],
                ),
                ui.output_ui("show_warning"),
                ui.output_ui("show_lexical_count"),
                ui.output_data_frame("get_search_results"),
                ui.output_ui("show_details_for_selected_rows"),
                style="display: flex; flex-direction: column; gap: 0.5rem; align-items: flex-start;",
            ),
            style=f"background:{UI_COLORS['search']} !important;",
            height="auto",
        ),
        # Right column for Chat.
        ui.card(
            ui.input_text_area(
                "chat_query",
                "Gib hier deine Frage oder Prompt ein:",
                value="",
                width="100%",
                rows=2,
            ),
            ui.input_task_button("chat_btn", "Fragen", width="25%", class_="btn-sm btn-success"),
            ui.output_ui("btn_click_warning"),
            ui.output_ui("result"),
            style=f"background:{UI_COLORS['chat']} !important;",
            height="auto",
        ),
        col_widths=[6, 6],
        row_heights=None,
    ),
)

# ---------------------------------------------------------------
# Server


def server(input, output, session):
    # ---------------------------------------------------------------
    # Settings and general reactive values
    hybrid_balance = reactive.Value(HYBRID_BALANCE)
    search_results = reactive.Value(None)
    lexical_results = reactive.Value(None)
    selected_search_results = reactive.Value(None)

    # If an input is triggered, update the settings.
    # Reactive effects re-execute when the dependencies change.
    # However, they do not return a value but rather update values or call functions.
    @reactive.effect
    def set_search_settings():
        hybrid_balance.set(input.hybrid_balance())

    @reactive.effect
    @reactive.event(input.show_appinfo)
    def _():
        m = ui.modal(
            ui.markdown(INFO_TEXT),
            ui.modal_button("Ok"),
            easy_close=True,
            footer=None,
            size="l",
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.show_instructions)
    def _():
        m = ui.modal(
            ui.markdown(INSTRUCTIONS),
            ui.modal_button("Ok"),
            easy_close=True,
            footer=None,
            size="l",
        )
        ui.modal_show(m)

    # ---------------------------------------------------------------
    # Search

    # Reactive events only fire when the input event is triggered,
    # not when dependencies change.
    @render.ui
    @reactive.event(input.search_btn)
    def show_warning():
        if not input.search_query():
            return ui.div("Bitte gib eine Suchanfrage ein.", class_="alert alert-warning")
        return ""

    @reactive.effect
    @reactive.event(input.copy_btn)
    def copy_search_to_chat():
        ui.update_text_area("chat_query", value=input.search_query())

    @render.data_frame
    @reactive.event(input.search_btn)
    def get_search_results():
        search_query = input.search_query()
        search_results.set(None)
        if search_query and search_query.strip() != "":
            ranked_index, result_chunks, cnt_bm25 = retrieve_ranked_chunks(
                search_query,
                hybrid_balance,
            )

            lexical_results.set(cnt_bm25)

            # Guard against stale Weaviate identifiers not present in the DataFrame.
            valid_ids = set(df["identifier"])
            paired = [
                (idx, chunk)
                for idx, chunk in zip(ranked_index, result_chunks)
                if idx in valid_ids
            ]
            if not paired:
                return
            valid_index, valid_chunks = zip(*paired)

            df_search_results = df.set_index("identifier").loc[list(valid_index)].reset_index()
            df_search_results["chunks"] = list(valid_chunks)
            search_results.set(df_search_results)

            display = df_search_results[["title", "token_count"]].copy()
            display.columns = ["Titel", "Tokens"]
            return render.DataGrid(
                display,
                selection_mode="rows",
                summary=False,
                width=800,
                height=600,
                # filters=True,
            )

    # Render UI is executed when any of the reactive values change.
    @render.ui
    def show_lexical_count():
        search_query = input.search_query()
        if lexical_results.get() == 0 and search_query.strip() != "":
            return ui.div(
                "Keine Entscheide über die lexikalische Suche mit dem exakten Stichwort gefunden.",
                class_="alert alert-warning",
            )
        return ""

    # Render UI is executed when any of the reactive values change.
    @render.ui
    def show_details_for_selected_rows():
        search_query = input.search_query()
        if input.get_search_results_selected_rows() and search_query.strip() != "":
            selected_search_results.set(
                search_results.get().loc[list(input.get_search_results_selected_rows())]
            )
            row = selected_search_results.get().iloc[0]
            if len(selected_search_results.get()) == 1:
                download_link = (
                    f'<p><a href={row.link} target="_blank">Link zu Originaldokument</a></p>'
                )
                text = f"<p><small><small>Relevanter Textabschnitt aus Suchtreffer:<br><small>{row.chunks}</small></small></p>"
                return ui.HTML(download_link + text)
            else:
                return ui.HTML(
                    f"<p><small>{len(selected_search_results.get())} Quellen ausgewählt.</small></p>"
                )
        else:
            if selected_search_results.get() is not None and search_results.get() is not None:
                return ui.HTML(
                    f"<p><small>Total {len(search_results.get())} gefunden. Wähle eins oder mehrere Dokumente aus.</small></p>"
                )
        return ""

    # ---------------------------------------------------------------
    # Chat
    @ui.bind_task_button(button_id="chat_btn")
    @reactive.extended_task
    async def call_llm(query, selected_search_results, model_choice):
        result = await chat_with_decisions(query, selected_search_results, model_choice)
        return result

    @reactive.effect
    @reactive.event(input.chat_btn)
    def btn_click():
        # Only call LLM if both search results are selected AND query is not empty
        if selected_search_results.get() is not None and input.chat_query().strip() != "":
            call_llm(
                input.chat_query(),
                selected_search_results.get(),
                input.model_choice(),
            )
        # If validation fails, the warning will be shown by btn_click_warning()

    @render.ui
    @reactive.event(input.chat_btn)
    def btn_click_warning():
        if selected_search_results.get() is None:
            return ui.div(
                "⚠️ Bitte wähle zuerst einen oder mehrere Entscheide aus den Suchresultaten aus.",
                class_="alert alert-warning",
            )
        if input.chat_query().strip() == "":
            return ui.div(
                "⚠️ Bitte gib eine Frage oder einen Prompt ein.",
                class_="alert alert-warning",
            )
        # Return empty string when validation passes to avoid empty space
        return ""

    @render.ui
    def result():
        result_value = call_llm.result()
        if result_value is None:
            return ""
        return ui.HTML(result_value)


# ---------------------------------------------------------------
# App

app = App(app_ui, server)
