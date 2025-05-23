import spacy
import tiktoken

nlp = spacy.load("de_core_news_lg")
tokenizer = tiktoken.encoding_for_model("gpt-4o")


def chunk_text(data, max_token_count=1000, overlap_tokens=200):
    """Chunk text into parts of max_token_count tokens with overlap_tokens tokens overlap.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    max_token_count : int, optional
        The maximum number of tokens per chunk, by default 1000.
    overlap_tokens : int, optional
        The number of tokens to overlap between chunks, by default 200.

    Returns
    -------
    list
        List of tuples containing the identifier and the chunked text.
    """

    # Sentencize text.
    doc = nlp(data.text)
    sents = [sent.text for sent in doc.sents]

    # Count tokens in each sentence.
    # TODO: Sentences can potentially be longer than max_token_count. Find a way to handle this.
    tokens = [len(tokenizer.encode(sent)) for sent in sents]

    # Create chunks by adding full sentences until max_token_count is reached.
    chunks = []
    current_chunk_start = 0
    current_sent = 0
    current_chunk = []
    current_tokens = 0

    while True:
        if current_sent >= len(sents):
            chunks.append(" ".join(current_chunk))
            break

        current_tokens += tokens[current_sent]
        if current_tokens < max_token_count:
            current_chunk.append(sents[current_sent])
            current_sent += 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0

            # Go back n sents until we create an overlap of overlap_tokens or more.
            count_back_tokens = 0
            count_back_sents = 0
            while True:
                count_back_tokens += tokens[current_sent]
                count_back_sents += 1
                if count_back_tokens > overlap_tokens:
                    break
                current_sent -= 1
            current_sent -= count_back_sents

            # Avoid endless loop if overlap_sents is too high.
            if current_sent <= current_chunk_start:
                current_sent = current_chunk_start + 1
            current_chunk_start = current_sent

    return [(data.identifier, chunk) for chunk in chunks]
