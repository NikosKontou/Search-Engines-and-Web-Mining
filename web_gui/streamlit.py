import os
import re
import json
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# --- Configuration & Constants ---
PAGE_TITLE = "Manga & TV Assistant"
DATA_DIRECTORY = "corpus"
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

def load_texts_from_directory(base_dir: str) -> Dict[str, Dict]:
    """
    Recursively loads JSON files from a directory.
    Returns a dict: {file_path: JSON_DATA}
    """
    docs_data = {}
    if not os.path.exists(base_dir):
        st.error(f"Directory not found: {base_dir}")
        return docs_data

    # print(f"Scanning directory: {base_dir}")
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # We specifically want the transformers text for RAG
                        text = data.get("transformers_text", "").strip()
                        if text:
                            docs_data[file_path] = data
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")
    return docs_data


def split_text_into_sentences(text: str) -> List[str]:
    pattern = r'(?<=[.?!;:])\s+|\n'
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def create_rolling_chunks(sentences: List[str], min_window: int, max_window: int, start_idx: int) -> Tuple[
    List[str], List[List[int]]]:
    doc_chunks = []
    doc_chunk_ids = []
    n_sentences = len(sentences)

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
    if 'rag_docs' not in st.session_state or not st.session_state['rag_docs']:
        st.warning("No documents loaded.")
        return

    with st.spinner("Processing RAG documents..."):
        all_sentences = []
        all_chunks = []
        all_chunk_ids = []
        doc_path_map = []  # Maps chunk index to document path

        # 1. Process Text (Extract text from loaded JSON data)
        for doc_path, doc_data in st.session_state['rag_docs'].items():
            text_content = doc_data.get("transformers_text", "")
            sentences = split_text_into_sentences(text_content)
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

            st.session_state['all_sentences'] = all_sentences
            st.session_state['rag_chunks'] = all_chunks
            st.session_state['rag_chunk_ids'] = all_chunk_ids
            st.session_state['chunk_doc_paths'] = doc_path_map
            st.session_state['doc_embeddings'] = np.array(embeddings)

            st.toast(f"Encoded {len(all_chunks)} chunks from {len(st.session_state['rag_docs'])} documents!")
        else:
            st.error("No valid text chunks found to embed.")


def find_similar_context(query: str) -> Dict[str, Any]:
    if st.session_state.get('doc_embeddings') is None:
        return {'indices': [], 'max_sim': 0.0, 'sentences': [], 'sources': set()}

    model = st.session_state['embeddings_model']
    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, st.session_state['doc_embeddings']).flatten()
    if similarities.size == 0:
        return {'indices': [], 'max_sim': 0.0, 'sentences': [], 'sources': set()}

    sorted_indices = similarities.argsort()[::-1]

    selected_chunk_indices = []
    selected_sentence_indices = set()
    found_sources = set()

    max_sim = float(similarities[sorted_indices[0]]) if sorted_indices.size > 0 else 0.0

    for idx in sorted_indices:
        if len(selected_sentence_indices) >= st.session_state['nof_keep_sentences']:
            break

        # Track sources (URLs)
        path = st.session_state['chunk_doc_paths'][idx]
        if path in st.session_state['rag_docs']:
            url = st.session_state['rag_docs'][path].get("url", "Unknown")
            found_sources.add(url)

        chunk_sentence_ids = st.session_state['rag_chunk_ids'][idx]
        selected_chunk_indices.append(int(idx))
        selected_sentence_indices.update(chunk_sentence_ids)

    return {
        'chunk_indices': selected_chunk_indices,
        'sentence_indices': sorted(list(selected_sentence_indices)),
        'max_similarity': max_sim,
        'sources': found_sources
    }


# --- LLM Interaction ---

def get_hf_client():
    token = os.getenv("HF_TOKEN")
    if st.session_state.get('space_id'):
        token = None
    return InferenceClient(st.session_state['llm_model_name'], token=token)


def query_llm(messages: List[Dict], max_tokens: int = 512) -> str:
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


def generate_rag_response(query: str, sentence_indices: List[int], sources: set) -> Tuple[str, Dict]:
    if not sentence_indices:
        return "No relevant information found.", {}

    MAX_CONTEXT_CHARS = 120000
    context_parts = []
    current_char_count = 0

    for idx in sentence_indices:
        sentence = st.session_state['all_sentences'][idx]
        if current_char_count + len(sentence) < MAX_CONTEXT_CHARS:
            context_parts.append(sentence)
            current_char_count += len(sentence)
        else:
            break

    context_text = "\n".join(context_parts)

    augmented_prompt = (
        f"Context information:\n\n{context_text}\n\n"
        f"Based on the above context, answer this question: {query}\n"
        "If the context doesn't contain relevant information, say you don't know based on the available information."
    )

    messages = st.session_state['chat_history'] + [{"role": "user", "content": augmented_prompt}]
    response = query_llm(messages, max_tokens=1024)

    # NOTE: Sources are now displayed in the sidebar, so we don't append them to the text here.

    retrieval_meta = {
        "context_length_chars": len(context_text),
        "sentences_retrieved": len(sentence_indices),
        "sources_count": len(sources)
    }
    return response, retrieval_meta


def run_hyde_process(query: str) -> Tuple[str, str, Dict, float]:
    hyde_messages = [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    hypothetical_answer = query_llm(hyde_messages)
    sim_results = find_similar_context(hypothetical_answer)

    if sim_results['max_similarity'] > st.session_state['similarity_threshold'] and sim_results['sentence_indices']:
        final_response, _ = generate_rag_response(query, sim_results['sentence_indices'], sim_results['sources'])
    else:
        final_response = (
            f"HyDE couldn't find relevant information. Similarity ({sim_results['max_similarity']:.2f}) "
            f"is below threshold ({st.session_state['similarity_threshold']})."
        )
    return final_response, hypothetical_answer, sim_results, sim_results['max_similarity']


# --- Session State Management ---

def initialize_session_state():
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
        'chunk_doc_paths': [],
        # NEW STATE VARIABLES FOR SIDEBAR
        'last_std_sources': set(),
        'last_hyde_sources': set(),
        'last_hyde_hypothetical': ""
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state['rag_docs'] is None:
        st.session_state['rag_docs'] = load_texts_from_directory(DATA_DIRECTORY)


# --- UI Components ---

def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        if st.button("Clear History"):
            st.session_state['chat_history'] = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.session_state['hyde_history'] = []
            st.session_state['last_std_sources'] = set()
            st.session_state['last_hyde_sources'] = set()
            st.session_state['last_hyde_hypothetical'] = ""
            st.rerun()

        st.divider()
        st.header("Inspection")

        # Display Standard RAG Sources
        st.subheader("Standard RAG Sources")
        if st.session_state['last_std_sources']:
            for url in st.session_state['last_std_sources']:
                st.markdown(f"- [{url}]({url})")
        else:
            st.caption("No sources available yet.")

        st.divider()

        # Display HyDE RAG Sources
        st.subheader("HyDE RAG Sources")
        if st.session_state['last_hyde_sources']:
            for url in st.session_state['last_hyde_sources']:
                st.markdown(f"- [{url}]({url})")
        else:
            st.caption("No HyDE sources available yet.")

        st.divider()

        # Display HyDE Hypothetical
        st.subheader("HyDE Hypothetical")
        if st.session_state['last_hyde_hypothetical']:
            with st.expander("Show Hypothetical Doc", expanded=True):
                st.markdown(st.session_state['last_hyde_hypothetical'])
        else:
            st.caption("No hypothetical document generated yet.")


def render_chat_interface():
    container = st.container(height=500)
    container.chat_message("ai", avatar=":material/robot_2:").markdown("Hello, how can I help you today?")
    for msg in st.session_state['chat_history']:
        if msg['role'] == "user":
            container.chat_message("user", avatar=":material/psychology_alt:").markdown(msg['content'])
        elif msg['role'] == "assistant":
            if msg.get('type') == 'hyde':
                with container.expander("🔍 **HyDE Response**"):
                    st.markdown(msg['content'])
            elif msg.get('type') == 'normal':
                container.chat_message("ai", avatar=":material/robot_2:").markdown(
                    f"**Standard RAG:** {msg['content']}")
    return container


# --- Main Application Execution ---

initialize_session_state()

if st.session_state['doc_embeddings'] is None:
    process_documents_and_embed()

render_sidebar()
msg_container = render_chat_interface()

if prompt := st.chat_input("Ask a question..."):
    msg_container.chat_message("user", avatar=":material/psychology_alt:").markdown(prompt)

    col1, col2 = msg_container.columns(2)

    with col1:
        with st.spinner("Standard RAG..."):
            sim_results = find_similar_context(prompt)
            if sim_results['max_similarity'] > st.session_state['similarity_threshold']:
                std_response, _ = generate_rag_response(prompt, sim_results['sentence_indices'], sim_results['sources'])
            else:
                std_response = f"Low similarity ({sim_results['max_similarity']:.2f}). No relevant info found."
            st.markdown("### Standard RAG")
            st.markdown(std_response)

    with col2:
        with st.spinner("HyDE processing..."):
            hyde_response, hyde_hypothetical, hyde_sim_results, hyde_score = run_hyde_process(prompt)
            st.markdown("### HyDE Response")
            st.markdown(hyde_response)

    # Save history
    st.session_state['chat_history'].append({"role": "user", "content": prompt})
    st.session_state['chat_history'].append({"role": "assistant", "content": std_response, "type": "normal"})
    st.session_state['chat_history'].append({"role": "assistant", "content": hyde_response, "type": "hyde"})

    # Update Session State for Sidebar
    st.session_state['last_std_sources'] = sim_results.get('sources', set())
    st.session_state['last_hyde_sources'] = hyde_sim_results.get('sources', set())
    st.session_state['last_hyde_hypothetical'] = hyde_hypothetical

    # Rerun to update sidebar immediately
    st.rerun()