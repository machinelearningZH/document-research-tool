from shiny import App, reactive, render, ui
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

import os
import pandas as pd
import time
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import weaviate
import weaviate.classes as wvc
import tiktoken

from utils_prompts import BASE_PROMPT
from utils_config import *

# Suppress Hugginface warning about tokenizers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.basicConfig(
    filename="app.log",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


# ---------------------------------------------------------------
# Init

# Load the documents that we will submit to the LLM.
df = pd.read_parquet(DATA_DIR + DOCUMENT_PARQUET_FILE)

try:
    openai_async_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=OPEN_ROUTER_API_KEY
    )
except Exception as e:
    logging.error(f"Error initializing OpenAI client for OpenRouter access: {e}")


# ---------------------------------------------------------------
# Functions


def log_interaction(selected_rows_index, query, answer, model_choice, success, start_time):
    """Log interaction."""
    end_time = time.time()
    logging.info(
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{selected_rows_index}\t{query}\t{answer}\t{model_choice}\t{success}\t{end_time - start_time:.3f}"
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


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = len(encoding.encode(string))
    return num_tokens


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
    query,
    hybrid_balance=HYBRID_BALANCE,
):
    """
    Retrieve relevant chunks from the data.

    Args:
        query (str): The search query
        hybrid_balance (reactive.Value): Balance between lexical and semantic search

    Returns:
        tuple: (result_index, result_chunks, response_bm25)
    """
    if not query or not query.strip():
        return [], [], 0

    try:
        embedding = embed_documents(query)
        if embedding is None:
            logging.error("Failed to generate embedding for query")
            return [], [], 0

        # We only perform BM25 search separately to get the count of lexical results
        # and inform the user if there are no lexical results.
        response_bm25 = collection.query.bm25(
            query=query,
            limit=10,
        )
        response_bm25_count = len(response_bm25.objects) if response_bm25.objects else 0

        # Perform the actual hybrid search.
        response = collection.query.hybrid(
            query=query,
            query_properties=["text"],
            vector=embedding,
            limit=300,
            alpha=hybrid_balance.get(),
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
    prompt,
    model_id="openai/gpt-4.1-mini",
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=8192,
):
    """
    Call the OpenAI API with appropriate parameters based on the model.

    Args:
        prompt (str): The prompt to send to the API
        model_id (str): Model identifier
        temperature (float): Temperature for generation
        max_tokens (int): Maximum tokens to generate

    Returns:
        str or None: Generated response or None on failure
    """
    if not prompt or not prompt.strip():
        logging.warning("Empty prompt provided to call_openai")
        return None

    try:
        # Common parameters.
        params = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Only add these parameters for models that support them.
        if model_id != "openai/o4-mini":
            params.update(
                {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )

        completion = await openai_async_client.chat.completions.create(**params)
        return completion.choices[0].message.content

    except Exception as e:
        logging.error(f"Error in call_openai: {str(e)}")
        return None


async def get_answer(query, context, model_choice):
    """
    Format prompt and get answer from API.

    Args:
        query (str): User query
        context (str): Context information
        model_choice (str): Model to use

    Returns:
        str or None: Model response or None on failure
    """
    if not query or not context:
        logging.warning("Missing query or context in get_answer")
        return None

    try:
        # Get the model key from the reverse mapping.
        model_key = MODEL_CHOICES_REVERSE.get(model_choice)
        if not model_key:
            logging.error(f"Unknown model choice: {model_choice}")
            return None

        # Get temperature for this model.
        temperature = MODEL_TEMPERATURES.get(model_key, DEFAULT_TEMPERATURE)

        prompt = BASE_PROMPT.format(context=context, question=query)
        return await call_openai(
            prompt,
            model_id=model_choice,
            temperature=temperature,
        )
    except Exception as e:
        logging.error(f"Error in get_answer: {str(e)}")
        return None


async def chat_with_decisions(query, selected_rows, model_choice):
    """
    Process user query with selected document rows.

    Args:
        query (str): User query
        selected_rows (DataFrame): Selected document rows
        model_choice (str): Model identifier

    Returns:
        str: Formatted response or error message
    """
    if not query or selected_rows is None or selected_rows.empty:
        return ui.markdown("Bitte gib eine Frage ein und wähle mindestens ein Dokument aus.")

    try:
        # Get model key from reverse mapping.
        model_key = MODEL_CHOICES_REVERSE.get(model_choice)
        if not model_key:
            return ui.markdown(f"Unbekanntes Modell: {model_choice}")

        # Create context from selected documents.
        context = "".join(
            [
                f"Quelle: {selected_rows.title.get(idx, 'Unbekannt')}\n{x.text}\n\n####################\n\n"
                for idx, x in selected_rows.iterrows()
            ]
        )

        # Check token count of composed context.
        num_tokens = num_tokens_from_string(context)
        # 7.8k plus Base Prompt will be around 8192 tokens, which is the smallest context length for small models.
        max_tokens = MAX_INPUT_TOKENS.get(model_key, 7_800)
        # Check if the number of tokens exceeds the limit.
        if num_tokens > max_tokens:
            return ui.markdown(
                f"Die ausgewählten Dokumente enthalten insgesamt **{num_tokens:,.0f} Tokens** und damit **zu viel Text für die Abfrage**. Bitte wähle weniger Dokumente aus.\n\nDie Limite betragen momentan: {', '.join(f'{v:,.0f} für {k}' for k, v in MAX_INPUT_TOKENS.items())}.\n\nBitte beachte, dass **zuviele Inhalte die Antwortqualität verschlechtern** und nicht verbessern. Es ist essentiell, möglichst wenige, treffende, relevante Inhalte auszuwählen und kein unnötiges «Informationsrauschen» an die Modelle zu schicken."
            )

        # Get answer from model.
        start_time = time.time()
        answer = await get_answer(query, context, model_choice)

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
        else:
            # Clean up and format answer.
            answer = answer.replace("```html", "").replace("```", "").strip()

            # Log successful interaction.
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
        style="background:#fafcff !important;",
    ),
    ui.layout_columns(
        # Left column for Search.
        ui.card(
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
            style="background:#fff8f5 !important;",
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
            style="background:#fafffb !important;",
        ),
    ),
)

# ---------------------------------------------------------------
# Server


def server(input, output, session):
    # ---------------------------------------------------------------
    # Settings and general reactive values
    hybrid_balance = reactive.Value(HYBRID_BALANCE)
    search_results = reactive.value(None)
    lexical_results = reactive.value(None)
    selected_search_results = reactive.value(None)

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
            df_search_results = df.set_index("identifier").loc[ranked_index].reset_index()
            df_search_results["chunks"] = result_chunks
            search_results.set(df_search_results)

            select_cols = ["title", "token_count"]
            display_cols = ["Titel", "Tokens"]
            display = df.set_index("identifier").loc[ranked_index].reset_index()[select_cols]
            display.columns = display_cols
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
                f"Keine Entscheide über die lexikalische Suche mit dem exakten Stichwort gefunden.",
                class_="alert alert-warning",
            )

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
        if selected_search_results.get() is not None and input.chat_query() != "":
            call_llm(
                input.chat_query(),
                selected_search_results.get(),
                input.model_choice(),
            )

    @render.ui
    @reactive.event(input.chat_btn)
    def btn_click_warning():
        if selected_search_results.get() is None:
            return ui.div(
                "Bitte wähle zuerst einen oder mehrere Entscheide aus.",
                class_="alert alert-warning",
            )
        if input.chat_query() == "":
            return ui.div(
                "Bitte gib eine Frage oder einen Prompt ein.",
                class_="alert alert-warning",
            )

    @render.ui
    def result():
        return ui.HTML(call_llm.result())


# ---------------------------------------------------------------
# App

# Create a Starlette app for static file serving.
# Use the StaticFiles middleware to serve static files from a local directory
# as a reference to user to validate the results.
# starlette_app = Starlette(
#     routes=[
#         Mount("/static", app=StaticFiles(directory=DOCS_DIR), name="static"),
#     ]
# )

starlette_app = Starlette()

# Create the Shiny app instance first.
shiny_app = App(app_ui, server)

# Mount the Shiny app onto the Starlette app.
starlette_app.mount("/", app=shiny_app)

# The final app object to be run is the Starlette app.
app = starlette_app
