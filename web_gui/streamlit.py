import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from huggingface_hub import InferenceClient
import os
import numpy as np

st.set_page_config(layout="wide")


def load_texts_from_directory(base_dir: str) -> dict:
    """Recursively read all .txt files from a directory and return {path: text}."""
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


# Load RAG documents
my_data_dir = "/Top/ACG/Year_2/timester 1/search engines/Search-Engines-and-Web-Mining/corpus"
rag_docs = load_texts_from_directory(my_data_dir)

# Combine for UI preview (optional)
my_initial_rag_text = "\n\n".join(
    f"--- {os.path.basename(path)} ---\n{text}" for path, text in rag_docs.items()
)

# Set the LLM model to meta-llama/Llama-3.1-8B-Instruct
if "my_llm_model" not in st.session_state:
    st.session_state['my_llm_model'] = "meta-llama/Llama-3.1-8B-Instruct"

if "my_space" not in st.session_state:
    st.session_state['my_space'] = os.environ.get("SPACE_ID")


def update_llm_model():
    """Initialize the InferenceClient for the Llama model"""
    if st.session_state['my_space']:
        st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'])
    else:
        st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'], token=os.getenv("HF_TOKEN"))


if "client" not in st.session_state:
    update_llm_model()

if "embeddings_model" not in st.session_state:
    st.session_state['embeddings_model'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

my_system_instructions = "You are a helpful assistant that answers questions about a manga and tv-series. Be brief and concise. Provide your answers in 100 words or less."

first_message = "Hello, how can I help you today?"

# NOTE: The chat delete function and RAG creation logic use keys for the removed sliders and text areas.
# I'll keep them for completeness, but they won't affect the UI as the widgets are removed.
def delete_chat_messages():
    for key in list(st.session_state.keys()):
        if key not in ["my_rag_text", "my_system_instructions", "my_llm_model", "my_space", "client",
                       "embeddings_model",
                       "my_sentences_rag", "my_sentences_rag_ids", "my_doc_ids", "my_sentences", "my_embeddings",
                       "min_window_size", "max_window_size", "my_similarity_threshold", "nof_keep_sentences"]: # Added necessary keys
            del st.session_state[key]
    update_llm_model()

# Placeholders for removed UI elements' default values
if "min_window_size" not in st.session_state:
    st.session_state["min_window_size"] = 5
if "max_window_size" not in st.session_state:
    st.session_state["max_window_size"] = 10
if "my_similarity_threshold" not in st.session_state:
    st.session_state["my_similarity_threshold"] = 0.2
if "nof_keep_sentences" not in st.session_state:
    st.session_state["nof_keep_sentences"] = 20
if "my_rag_text" not in st.session_state:
    st.session_state["my_rag_text"] = my_initial_rag_text


def create_sentences_rag():
    # Use st.empty() for rag_status_placeholder even though it won't be visible in the final requested UI
    rag_status_placeholder = st.empty()
    with rag_status_placeholder:
        st.info("Processing RAG documents...")
        pattern = r'(?<=[.?!;:])\s+|\n'

        # Reset all sentence and embedding state
        st.session_state['my_sentences_rag'] = []
        st.session_state['my_sentences_rag_ids'] = []
        st.session_state['my_doc_ids'] = []
        st.session_state['my_sentences'] = []
        st.session_state['my_embeddings'] = None

        all_embeddings = []

        # Loop over each document separately
        for doc_path, doc_text in rag_docs.items():
            sentences = [s.strip() for s in re.split(pattern, doc_text) if s.strip()]

            if not sentences:
                continue

            # Store individual sentences for this document
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
                        # Store global sentence indices for this chunk
                        global_indices = list(
                            range(doc_sentence_start_idx + i, doc_sentence_start_idx + i + rolling_window_size))
                        doc_chunk_ids.append(global_indices)
                        st.session_state['my_doc_ids'].append(doc_path)

            # Handle very short docs
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
    """Calculate similarities for a given prompt and return the most relevant chunks"""

    # DEBUG: Print what we're working with
    print(f"Number of chunks: {len(st.session_state['my_sentences_rag'])}")
    print(
        f"Embeddings shape: {st.session_state['my_embeddings'].shape if st.session_state['my_embeddings'] is not None else 'None'}")

    # Use the FULL prompt for similarity calculation (more reliable than sub-prompts)
    my_question_embedding = st.session_state.embeddings_model.encode([prompt])
    similarities = cosine_similarity(my_question_embedding, st.session_state['my_embeddings']).flatten()

    # Get top similar chunks
    sorted_indices = similarities.argsort()[::-1]  # Sort descending
    selected_chunk_indices = []
    selected_sentence_indices = set()
    max_similarity = similarities[sorted_indices[0]] if len(sorted_indices) > 0 else 0

    # Select top chunks until we have enough unique sentences
    for chunk_idx in sorted_indices:
        if len(selected_sentence_indices) >= st.session_state['nof_keep_sentences']:
            break

        chunk_sentence_indices = st.session_state['my_sentences_rag_ids'][chunk_idx]

        # Add this chunk to our selection
        selected_chunk_indices.append(chunk_idx)
        selected_sentence_indices.update(chunk_sentence_indices)

    selected_sentence_indices = sorted(list(selected_sentence_indices))

    return {
        'selected_chunk_indices': selected_chunk_indices,
        'selected_sentence_indices': selected_sentence_indices,
        'similarities': similarities,
        'max_similarity': max_similarity
    }


# --- START OF MODIFIED UI ---

# 1. Disclaimer dropdown
st.expander("Disclaimer", expanded=False).markdown("""This application and code (hereafter referred to as the 'Software') is a proof of concept at an experimental stage and is not intended to be used as a production environment. The Software is provided as is, wihtout any warranties of any kind, expressed or implied and the user assumes full responsibility for its use, implementation, and legal compliance.

The developers of the Software shall not be liable for any damages, losses, claims, or liabilities arising from the Software, including but not limited to the usage of artificial intelligence and machine learning, related errors, third-party tool failures, security breaches, intellectual property violations, legal or regulatory non-compliance, deployment risks, or any indirect, incidental, or consequential damages.

Large Language Models may provide wrong answers. Please verify the answers and comply with applicable laws and regulations.

The user agrees to indemnify and hold harmless the developers of the Software from any related claims or disputes arising from the utilization of the Software by the user.

By using the Software, you agree to the terms and conditions of the disclaimer.""")

# Initialize chat messages and RAG data if needed
if "my_chat_messages" not in st.session_state:
    st.session_state['my_chat_messages'] = []
    st.session_state['my_chat_messages'].append(
        {"role": "system", "content": my_system_instructions})

if "my_sentences_rag" not in st.session_state:
    create_sentences_rag()

# 2. Chat box
messages_container = st.container(height=500)

messages_container.chat_message("ai", avatar=":material/robot_2:").markdown(first_message)

for message in st.session_state['my_chat_messages']:
    if message['role'] == "user":
        messages_container.chat_message(message['role'], avatar=":material/psychology_alt:").markdown(
            message['content'])
    elif message['role'] == "assistant":
        messages_container.chat_message(message['role'], avatar=":material/robot_2:").markdown(message['content'])

if prompt := st.chat_input("you may ask here your questions"):
    # Add user message immediately
    messages_container.chat_message("user", avatar=":material/psychology_alt:").markdown(prompt)

    with messages_container.chat_message("ai", avatar=":material/robot_2:"):
        response_placeholder = st.empty()

        # Show that we're processing
        response_placeholder.info("Searching for relevant information...")

        # Recalculate similarities for EVERY new prompt
        similarity_results = find_most_similar_chunks(prompt)

        selected_chunk_indices = similarity_results['selected_chunk_indices']
        selected_sentence_indices = similarity_results['selected_sentence_indices']
        similarities = similarity_results['similarities']
        max_similarity = similarity_results['max_similarity']

        # --- BEGIN: Removed retrieval preview and chunk details (bottom_col1) ---
        # The logic to generate 'response' still relies on these variables
        # so they must be calculated above, but the display is removed here.
        # --- END: Removed retrieval preview and chunk details ---

        # Generate response
        if max_similarity > st.session_state['my_similarity_threshold'] and selected_sentence_indices:
            # Clear the processing message
            response_placeholder.empty()

            augmented_prompt = "Context information:\n\n"
            context_text = "\n".join([st.session_state['my_sentences'][idx] for idx in selected_sentence_indices])
            augmented_prompt += context_text
            augmented_prompt += "\n\nBased on the above context, answer this question: " + prompt
            augmented_prompt += "\nIf the context doesn't contain relevant information, say you don't know based on the available information."

            temp_messages = st.session_state['my_chat_messages'] + [{"role": "user", "content": augmented_prompt}]

            response = ""
            try:
                for chunk in st.session_state['client'].chat.completions.create(
                        messages=temp_messages,
                        model=st.session_state['my_llm_model'],
                        stream=True,
                        max_tokens=1024):
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
                        response_placeholder.markdown(response)
            except Exception as e:
                response = f"Error generating response: {str(e)}"
                response_placeholder.markdown(response)
        else:
            response = f"I don't have enough relevant information to answer this question. The best match has {100 * max_similarity:.2f}% similarity, which is below the threshold of {100 * st.session_state['my_similarity_threshold']:.2f}%."
            response_placeholder.markdown(response)

    # Store conversation
    st.session_state['my_chat_messages'].append({"role": "user", "content": prompt})
    st.session_state['my_chat_messages'].append({"role": "assistant", "content": response})

    # Limit chat history
    if len(st.session_state['my_chat_messages']) > 10:
        st.session_state['my_chat_messages'] = [st.session_state['my_chat_messages'][0]] + st.session_state[
            'my_chat_messages'][-8:]


    # 1. Get a list of unique document file names (sources) for the retrieved chunks
    retrieved_doc_paths = [st.session_state['my_doc_ids'][i] for i in selected_chunk_indices]
    unique_doc_names = sorted(list(set(os.path.basename(path) for path in retrieved_doc_paths)))

    # 2. Create the retrieval info dictionary
    retrieval_info = {
        "query": prompt,
        "max_similarity": float(max_similarity),
        "threshold": float(st.session_state['my_similarity_threshold']),
        "chunks_retrieved": len(selected_chunk_indices),
        "sentences_used": len(selected_sentence_indices),
        "total_chunks_available": len(st.session_state['my_sentences_rag']),
        "similar_documents_found": unique_doc_names  # ADDED: List of unique document names
    }

    # 3. Display the Retrieval Information
    with st.expander("Retrieval Info"): # Display under the chat for a single column layout
        st.write("**Retrieval Information for Current Query:**")
        st.json(retrieval_info, expanded=True)