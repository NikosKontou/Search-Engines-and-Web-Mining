import os
import re
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple, Set, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# --- Configuration & Constants ---
PAGE_TITLE = "Manga & TV Assistant"
DATA_DIRECTORY = "/Top/ACG/Year_2/timester 1/search engines/Search-Engines-and-Web-Mining/corpus_transform"
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# System Instructions
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a manga and tv-series. "
    "Be brief and concise. Provide your answers in 100 words or less."
)
HYDE_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates a hypothetical answer to the user's question. "
    "Be brief and concise. Provide your answer in 100 words or less."
)

st.set_page_config(page_title=PAGE_TITLE, layout="wide")


# --- Helper Functions: Data Loading & Processing ---

def load_texts_from_directory(base_dir: str) -> Dict[str, str]:
    """Recursively loads text files from a directory."""
    texts = {}
    if not os.path.exists(base_dir):
        st.error(f"Directory not found: {base_dir}")
        return texts

    print(f"Scanning directory: {base_dir}")
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if text:
                            texts[file_path] = text
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
    return texts


def split_text_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using regex."""
    pattern = r'(?<=[.?!;:])\s+|\n'
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def create_rolling_chunks(sentences: List[str], min_window: int, max_window: int, start_idx: int) -> Tuple[
    List[str], List[List[int]]]:
    """Creates text chunks using a rolling window approach."""
    doc_chunks = []
    doc_chunk_ids = []

    n_sentences = len(sentences)
    # Handle very short documents
    if n_sentences < min_window:
        chunk = " ".join(sentences)
        doc_chunks.append(chunk)
        doc_chunk_ids.append(list(range(start_idx, start_idx + n_sentences)))
        return doc_chunks, doc_chunk_ids

    actual_max = min(max_window, n_sentences)
    actual_min = min(min_window, actual_max)

    for window_size in range(actual_min, actual_max + 1):
        for i in range(n_sentences - window_size + 1):
            chunk = " ".join(sentences[i: i + window_size]).strip()
            if chunk:
                doc_chunks.append(chunk)
                global_indices = list(range(start_idx + i, start_idx + i + window_size))
                doc_chunk_ids.append(global_indices)

    return doc_chunks, doc_chunk_ids


# --- Core Logic: Embeddings & Search ---

def process_documents_and_embed():
    """Processes loaded documents, chunks them, and generates embeddings."""
    if 'rag_docs' not in st.session_state or not st.session_state['rag_docs']:
        st.warning("No documents loaded.")
        return

    with st.spinner("Processing RAG documents..."):
        all_sentences = []
        all_chunks = []
        all_chunk_ids = []  # List of list of sentence indices
        doc_path_map = []  # Maps chunk index to document path

        # 1. Process Text
        for doc_path, doc_text in st.session_state['rag_docs'].items():
            sentences = split_text_into_sentences(doc_text)
            if not sentences:
                continue

            current_sentence_idx = len(all_sentences)
            all_sentences.extend(sentences)

            chunks, chunk_ids = create_rolling_chunks(
                sentences,
                st.session_state['min_window_size'],
                st.session_state['max_window_size'],
                current_sentence_idx
            )

            all_chunks.extend(chunks)
            all_chunk_ids.extend(chunk_ids)
            doc_path_map.extend([doc_path] * len(chunks))

        # 2. Generate Embeddings
        if all_chunks:
            model = st.session_state['embeddings_model']
            embeddings = model.encode(all_chunks)

            # Update Session State
            st.session_state['all_sentences'] = all_sentences
            st.session_state['rag_chunks'] = all_chunks
            st.session_state['rag_chunk_ids'] = all_chunk_ids
            st.session_state['chunk_doc_paths'] = doc_path_map
            st.session_state['doc_embeddings'] = np.array(embeddings)

            st.success(f"Encoded {len(all_chunks)} chunks from {len(st.session_state['rag_docs'])} documents!")
        else:
            st.error("No valid text chunks found to embed.")


def find_similar_context(query: str) -> Dict[str, Any]:
    """Finds most similar chunks to the query."""
    if st.session_state.get('doc_embeddings') is None:
        return {'indices': [], 'max_sim': 0.0, 'sentences': []}

    model = st.session_state['embeddings_model']
    query_embedding = model.encode([query])

    # Calculate Cosine Similarity
    similarities = cosine_similarity(query_embedding, st.session_state['doc_embeddings']).flatten()

    if similarities.size == 0:
        return {'indices': [], 'max_sim': 0.0, 'sentences': []}

    # Sort results
    sorted_indices = similarities.argsort()[::-1]

    selected_chunk_indices = []
    selected_sentence_indices = set()
    max_sim = float(similarities[sorted_indices[0]]) if sorted_indices.size > 0 else 0.0

    # Select chunks until sentence limit reached
    for idx in sorted_indices:
        if len(selected_sentence_indices) >= st.session_state['nof_keep_sentences']:
            break

        chunk_sentence_ids = st.session_state['rag_chunk_ids'][idx]
        selected_chunk_indices.append(int(idx))
        selected_sentence_indices.update(chunk_sentence_ids)

    return {
        'chunk_indices': selected_chunk_indices,
        'sentence_indices': sorted(list(selected_sentence_indices)),
        'max_similarity': max_sim
    }


# --- LLM Interaction ---

def get_hf_client():
    """Initializes or retrieves the HuggingFace Inference Client."""
    token = os.getenv("HF_TOKEN")
    if st.session_state.get('space_id'):
        token = None  # Spaces often handle auth internally or via specific env vars
    return InferenceClient(st.session_state['llm_model_name'], token=token)


def query_llm(messages: List[Dict], max_tokens: int = 512) -> str:
    """Generic function to stream responses from the LLM."""
    client = get_hf_client()
    response_text = ""
    try:
        stream = client.chat.completions.create(
            messages=messages,
            model=st.session_state['llm_model_name'],
            stream=True,
            max_tokens=max_tokens
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                response_text += content
        return response_text.strip()
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"


def generate_rag_response(query: str, sentence_indices: List[int]) -> Tuple[str, Dict]:
    """Generates a response based on retrieved context with token safety limits."""
    if not sentence_indices:
        return "No relevant information found.", {}

    # --- SAFETY FIX START ---
    # Heuristic: 1 token ~= 4 chars.
    # Limit context to ~30,000 tokens (approx 120,000 chars) to be safe and leave room for history/response.
    MAX_CONTEXT_CHARS = 120000

    context_parts = []
    current_char_count = 0

    # Iterate through indices and add text until we hit the limit
    for idx in sentence_indices:
        sentence = st.session_state['all_sentences'][idx]
        if current_char_count + len(sentence) < MAX_CONTEXT_CHARS:
            context_parts.append(sentence)
            current_char_count += len(sentence)
        else:
            # Stop adding context if we exceed the limit
            break

    context_text = "\n".join(context_parts)
    # --- SAFETY FIX END ---

    augmented_prompt = (
        f"Context information:\n\n{context_text}\n\n"
        f"Based on the above context, answer this question: {query}\n"
        "If the context doesn't contain relevant information, say you don't know based on the available information."
    )

    # Use existing chat history + new prompt
    messages = st.session_state['chat_history'] + [{"role": "user", "content": augmented_prompt}]

    # (Optional) Double check total history size here if needed

    response = query_llm(messages, max_tokens=1024)

    retrieval_meta = {
        "context_length_chars": len(context_text),
        "sentences_retrieved": len(sentence_indices),
        "sentences_used": len(context_parts)  # Might be lower if truncation occurred
    }
    return response, retrieval_meta

def run_hyde_process(query: str) -> Tuple[str, str, Dict, float]:
    """Runs the HyDE (Hypothetical Document Embeddings) pipeline."""
    # 1. Generate Hypothetical Answer
    hyde_messages = [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    hypothetical_answer = query_llm(hyde_messages)

    # 2. Retrieve based on Hypothetical Answer
    sim_results = find_similar_context(hypothetical_answer)

    # 3. Generate Final Response
    if sim_results['max_similarity'] > st.session_state['similarity_threshold'] and sim_results['sentence_indices']:
        final_response, _ = generate_rag_response(query, sim_results['sentence_indices'])
    else:
        final_response = (
            f"HyDE couldn't find relevant information. Similarity ({sim_results['max_similarity']:.2f}) "
            f"is below threshold ({st.session_state['similarity_threshold']})."
        )

    return final_response, hypothetical_answer, sim_results, sim_results['max_similarity']


# --- Session State Management ---

def initialize_session_state():
    """Initializes all session state variables."""
    defaults = {
        'llm_model_name': DEFAULT_LLM_MODEL,
        'space_id': os.environ.get("SPACE_ID"),
        'embeddings_model': SentenceTransformer(EMBEDDING_MODEL_NAME),
        'min_window_size': 5,
        'max_window_size': 10,
        'similarity_threshold': 0.2,
        'nof_keep_sentences': 20,
        'chat_history': [{"role": "system", "content": SYSTEM_PROMPT}],
        'hyde_history': [],
        'rag_docs': None,
        'all_sentences': [],
        'doc_embeddings': None,
        'current_comparison': {}
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Load Docs only once
    if st.session_state['rag_docs'] is None:
        st.session_state['rag_docs'] = load_texts_from_directory(DATA_DIRECTORY)


# --- UI Components ---

def render_sidebar():
    with st.sidebar:
        st.header("HyDE Settings")

        if st.session_state['hyde_history']:
            st.subheader("Recent HyDE Responses")
            # Show last 5
            for entry in reversed(st.session_state['hyde_history'][-5:]):
                with st.expander(f"Q: {entry['question'][:40]}..."):
                    st.caption("Hypothetical Answer:")
                    st.write(entry['hypothetical'][:150] + "...")
                    st.caption("Final Result:")
                    st.write(entry['response'])

        if st.button("Clear HyDE History"):
            st.session_state['hyde_history'] = []
            st.rerun()


def render_chat_interface():
    container = st.container(height=500)
    container.chat_message("ai", avatar=":material/robot_2:").markdown("Hello, how can I help you today?")

    # Display History
    for msg in st.session_state['chat_history']:
        if msg['role'] == "user":
            container.chat_message("user", avatar=":material/psychology_alt:").markdown(msg['content'])
        elif msg['role'] == "assistant":
            # We differentiate stored messages by checking if they are 'normal' or 'hyde' in a custom key
            # or simply display standard chat history.
            # For this specific app structure, we display the last generated results explicitly below.
            # To avoid clutter, we skip rendering old standard messages inside the loop if the logic relies on the
            # 'current_comparison' view, but for a standard chat feel:
            if msg.get('type') == 'hyde':
                with container.expander("🔍 **HyDE Response**"):
                    st.markdown(msg['content'])
            elif msg.get('type') == 'normal':
                container.chat_message("ai", avatar=":material/robot_2:").markdown(
                    f"**Standard RAG:** {msg['content']}")

    return container


# --- Main Application Execution ---

initialize_session_state()

# Ensure embeddings exist
if st.session_state['doc_embeddings'] is None:
    process_documents_and_embed()

# Disclaimer
st.expander("Disclaimer").markdown(
    "This application is experimental. Large Language Models may provide wrong answers."
)

render_sidebar()
msg_container = render_chat_interface()

# Input Handling
if prompt := st.chat_input("Ask a question regarding the documents..."):
    msg_container.chat_message("user", avatar=":material/psychology_alt:").markdown(prompt)

    col1, col2 = msg_container.columns(2)

    # --- Standard RAG ---
    with col1:
        with st.spinner("Standard RAG..."):
            st.markdown("### Standard RAG")
            sim_results = find_similar_context(prompt)

            if sim_results['max_similarity'] > st.session_state['similarity_threshold']:
                std_response, _ = generate_rag_response(prompt, sim_results['sentence_indices'])
            else:
                std_response = f"Low similarity ({sim_results['max_similarity']:.2f}). No relevant info found."

            st.markdown(std_response)

            # Metadata for display
            std_meta = {
                "max_similarity": float(sim_results['max_similarity']),
                "chunks_found": len(sim_results['chunk_indices'])
            }

    # --- HyDE RAG ---
    with col2:
        with st.spinner("HyDE processing..."):
            st.markdown("### HyDE Response")
            hyde_response, hyde_hypothetical, hyde_sim_results, hyde_score = run_hyde_process(prompt)
            st.markdown(hyde_response)

            # Metadata for display
            hyde_meta = {
                "max_similarity": float(hyde_score),
                "hypothetical_preview": hyde_hypothetical[:100] + "..."
            }

    # Update History
    st.session_state['chat_history'].append({"role": "user", "content": prompt})
    st.session_state['chat_history'].append({"role": "assistant", "content": std_response, "type": "normal"})
    st.session_state['chat_history'].append({"role": "assistant", "content": hyde_response, "type": "hyde"})

    # Update HyDE specific history
    st.session_state['hyde_history'].append({
        "question": prompt,
        "hypothetical": hyde_hypothetical,
        "response": hyde_response
    })

    # Keep history manageable
    if len(st.session_state['chat_history']) > 20:
        st.session_state['chat_history'] = [st.session_state['chat_history'][0]] + st.session_state['chat_history'][
            -18:]

    # --- Debug/Info Expanders ---
    with msg_container.expander("Retrieval Details", expanded=False):
        t1, t2 = st.tabs(["Standard", "HyDE"])
        t1.json(std_meta)
        t2.json(hyde_meta)
        t2.markdown(f"**Full Hypothetical Answer:**\n{hyde_hypothetical}")

    # --- Comparison View ---
    with msg_container.expander("Compare Responses", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Standard")
            st.info(std_response)
        with c2:
            st.caption("HyDE")
            st.success(hyde_response)