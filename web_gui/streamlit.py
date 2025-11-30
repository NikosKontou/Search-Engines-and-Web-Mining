import os
import re
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

st.set_page_config(layout="wide")

def load_texts_from_directory(base_dir: str) -> dict:
    texts = {}
    print(f"current directory is {base_dir}")
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

# load RAG documents
my_data_dir = "/Top/ACG/Year_2/timester 1/search engines/Search-Engines-and-Web-Mining/corpus_transform"
rag_docs = load_texts_from_directory(my_data_dir)

# combine for UI preview
my_initial_rag_text = "\n\n".join(
    f"--- {os.path.basename(path)} ---\n{text}" for path, text in rag_docs.items()
)

# session defaults
st.session_state.setdefault('my_llm_model', "meta-llama/Llama-3.1-8B-Instruct")
st.session_state.setdefault('my_space', os.environ.get("SPACE_ID"))

def update_llm_model():
    token = None if st.session_state.get('my_space') else os.getenv("HF_TOKEN")
    st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'], token=token)

if "client" not in st.session_state:
    update_llm_model()

st.session_state.setdefault('embeddings_model', SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))

my_system_instructions = "You are a helpful assistant that answers questions about a manga and tv-series. Be brief and concise. Provide your answers in 100 words or less."
first_message = "Hello, how can I help you today?"

def delete_chat_messages():
    keep = {
        "my_rag_text", "my_system_instructions", "my_llm_model", "my_space", "client",
        "embeddings_model", "my_sentences_rag", "my_sentences_rag_ids", "my_doc_ids",
        "my_sentences", "my_embeddings", "min_window_size", "max_window_size",
        "my_similarity_threshold", "nof_keep_sentences"
    }
    for key in list(st.session_state.keys()):
        if key not in keep:
            del st.session_state[key]
    update_llm_model()

# placeholders / defaults for UI values removed earlier
st.session_state.setdefault("min_window_size", 5)
st.session_state.setdefault("max_window_size", 10)
st.session_state.setdefault("my_similarity_threshold", 0.2)
st.session_state.setdefault("nof_keep_sentences", 20)
st.session_state.setdefault("my_rag_text", my_initial_rag_text)

def create_sentences_rag():
    rag_status_placeholder = st.empty()
    with rag_status_placeholder:
        st.info("Processing RAG documents...")
        pattern = r'(?<=[.?!;:])\s+|\n'

        st.session_state['my_sentences_rag'] = []
        st.session_state['my_sentences_rag_ids'] = []
        st.session_state['my_doc_ids'] = []
        st.session_state['my_sentences'] = []
        st.session_state['my_embeddings'] = None

        all_embeddings = []

        for doc_path, doc_text in rag_docs.items():
            sentences = [s.strip() for s in re.split(pattern, doc_text) if s.strip()]
            if not sentences:
                continue

            doc_sentence_start_idx = len(st.session_state['my_sentences'])
            st.session_state['my_sentences'].extend(sentences)

            max_window = min(st.session_state['max_window_size'], len(sentences))
            min_window = min(st.session_state['min_window_size'], max_window)

            doc_chunks = []
            doc_chunk_ids = []

            for rolling_window_size in range(min_window, max_window + 1):
                if rolling_window_size > len(sentences):
                    continue
                for i in range(0, len(sentences) - rolling_window_size + 1):
                    chunk = " ".join(sentences[i:i + rolling_window_size]).strip()
                    if chunk:
                        doc_chunks.append(chunk)
                        global_indices = list(range(doc_sentence_start_idx + i,
                                                    doc_sentence_start_idx + i + rolling_window_size))
                        doc_chunk_ids.append(global_indices)
                        st.session_state['my_doc_ids'].append(doc_path)

            # handle very short docs
            if not doc_chunks and sentences:
                short_chunk = " ".join(sentences)
                doc_chunks = [short_chunk]
                global_indices = list(range(doc_sentence_start_idx, doc_sentence_start_idx + len(sentences)))
                doc_chunk_ids.append(global_indices)
                st.session_state['my_doc_ids'].append(doc_path)

            if doc_chunks:
                embeddings = st.session_state['embeddings_model'].encode(doc_chunks)
                all_embeddings.extend(embeddings)
                st.session_state['my_sentences_rag'].extend(doc_chunks)
                st.session_state['my_sentences_rag_ids'].extend(doc_chunk_ids)

        if all_embeddings:
            st.session_state['my_embeddings'] = np.array(all_embeddings)
            st.success(f"Encoded {len(st.session_state['my_sentences_rag'])} chunks from {len(rag_docs)} documents!")
        else:
            st.error("No embeddings were created!")

def find_most_similar_chunks(prompt):
    # guard if no embeddings yet
    if st.session_state.get('my_embeddings') is None:
        return {
            'selected_chunk_indices': [],
            'selected_sentence_indices': [],
            'similarities': np.array([]),
            'max_similarity': 0.0
        }

    my_question_embedding = st.session_state['embeddings_model'].encode([prompt])
    similarities = cosine_similarity(my_question_embedding, st.session_state['my_embeddings']).flatten()
    if similarities.size == 0:
        return {'selected_chunk_indices': [], 'selected_sentence_indices': [], 'similarities': similarities, 'max_similarity': 0.0}

    sorted_indices = similarities.argsort()[::-1]
    selected_chunk_indices = []
    selected_sentence_indices = set()
    max_similarity = float(similarities[sorted_indices[0]]) if sorted_indices.size > 0 else 0.0

    for chunk_idx in sorted_indices:
        if len(selected_sentence_indices) >= st.session_state['nof_keep_sentences']:
            break
        # guard in case rag ids list is shorter than expected
        if chunk_idx >= len(st.session_state.get('my_sentences_rag_ids', [])):
            continue
        chunk_sentence_indices = st.session_state['my_sentences_rag_ids'][chunk_idx]
        selected_chunk_indices.append(int(chunk_idx))
        selected_sentence_indices.update(chunk_sentence_indices)

    selected_sentence_indices = sorted(list(selected_sentence_indices))
    return {
        'selected_chunk_indices': selected_chunk_indices,
        'selected_sentence_indices': selected_sentence_indices,
        'similarities': similarities,
        'max_similarity': max_similarity
    }

# ui
st.expander("Disclaimer", expanded=False).markdown(
    "This application is experimental. Large Language Models may provide wrong answers. Verify results and comply with applicable laws."
)

if "my_chat_messages" not in st.session_state:
    st.session_state['my_chat_messages'] = [{"role": "system", "content": my_system_instructions}]

if "my_sentences_rag" not in st.session_state:
    create_sentences_rag()

messages_container = st.container(height=500)
messages_container.chat_message("ai", avatar=":material/robot_2:").markdown(first_message)

for message in st.session_state['my_chat_messages']:
    role = message.get('role')
    content = message.get('content', '')
    if role == "user":
        messages_container.chat_message("user", avatar=":material/psychology_alt:").markdown(content)
    elif role == "assistant":
        messages_container.chat_message("ai", avatar=":material/robot_2:").markdown(content)

if prompt := st.chat_input("you may ask here your questions"):
    messages_container.chat_message("user", avatar=":material/psychology_alt:").markdown(prompt)
    with messages_container.chat_message("ai", avatar=":material/robot_2:"):
        response_placeholder = st.empty()
        response_placeholder.info("Searching for relevant information...")

        similarity_results = find_most_similar_chunks(prompt)
        selected_chunk_indices = similarity_results['selected_chunk_indices']
        selected_sentence_indices = similarity_results['selected_sentence_indices']
        similarities = similarity_results['similarities']
        max_similarity = similarity_results['max_similarity']

        # generate response if we have good match
        if max_similarity > float(st.session_state['my_similarity_threshold']) and selected_sentence_indices:
            response_placeholder.empty()
            context_text = "\n".join([st.session_state['my_sentences'][idx] for idx in selected_sentence_indices])
            augmented_prompt = (
                "Context information:\n\n"
                + context_text
                + "\n\nBased on the above context, answer this question: "
                + prompt
                + "\nIf the context doesn't contain relevant information, say you don't know based on the available information."
            )

            temp_messages = st.session_state['my_chat_messages'] + [{"role": "user", "content": augmented_prompt}]

            response = ""
            try:
                for packet in st.session_state['client'].chat.completions.create(
                        messages=temp_messages,
                        model=st.session_state['my_llm_model'],
                        stream=True,
                        max_tokens=1024):
                    # safe extraction of delta content
                    choice = getattr(packet, "choices", None)
                    if not choice:
                        continue
                    first_choice = choice[0]
                    delta = getattr(first_choice, "delta", None)
                    # delta may be None or have no content
                    content_piece = getattr(delta, "content", None)
                    if not content_piece:
                        continue
                    response += content_piece
                    response_placeholder.markdown(response)
            except Exception as e:
                response = f"Error generating response: {str(e)}"
                response_placeholder.markdown(response)
        else:
            response = (
                f"I don't have enough relevant information to answer this question. "
                f"The best match has {100 * float(max_similarity):.2f}% similarity, which is below the threshold of "
                f"{100 * float(st.session_state['my_similarity_threshold']):.2f}%."
            )
            response_placeholder.markdown(response)

    # store conversation safely
    st.session_state['my_chat_messages'].append({"role": "user", "content": prompt})
    st.session_state['my_chat_messages'].append({"role": "assistant", "content": response})

    if len(st.session_state['my_chat_messages']) > 10:
        st.session_state['my_chat_messages'] = [st.session_state['my_chat_messages'][0]] + st.session_state['my_chat_messages'][-8:]

    # retrieval info and doc list for retrieved chunks (safe handling)
    retrieved_doc_paths = []
    for i in selected_chunk_indices:
        if i < len(st.session_state.get('my_doc_ids', [])):
            retrieved_doc_paths.append(st.session_state['my_doc_ids'][i])
    unique_doc_names = sorted(list({os.path.basename(p) for p in retrieved_doc_paths}))

    retrieval_info = {
        "query": prompt,
        "max_similarity": float(max_similarity),
        "threshold": float(st.session_state['my_similarity_threshold']),
        "chunks_retrieved": len(selected_chunk_indices),
        "sentences_used": len(selected_sentence_indices),
        "total_chunks_available": len(st.session_state.get('my_sentences_rag', [])),
        "similar_documents_found": unique_doc_names
    }

    with st.expander("Retrieval Info"):
        st.write("**Retrieval Information for Current Query:**")
        st.json(retrieval_info, expanded=True)
