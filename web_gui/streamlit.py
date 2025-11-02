import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from huggingface_hub import InferenceClient
import os
import numpy as np
from openai import OpenAI

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

# Check if the LLM model is not already in the session state
if "my_llm_model" not in st.session_state:
    st.session_state['my_llm_model'] = "mistralai/Mistral-7B-Instruct-v0.3"

if "my_space" not in st.session_state:
    st.session_state['my_space'] = os.environ.get("SPACE_ID")


def update_llm_model():
    if st.session_state['my_llm_model'].startswith("gemini-"):
        st.session_state['client'] = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif st.session_state['my_llm_model'].startswith("gpt-"):
        st.session_state['client'] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        if st.session_state['my_space']:
            st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'])
        else:
            st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'], token=os.getenv("HF_TOKEN"))


if "client" not in st.session_state:
    update_llm_model()

if "embeddings_model" not in st.session_state:
    st.session_state['embeddings_model'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

my_system_instructions = "You are a helpful assistant that answers questions about the manga and tv-series one piece. Be brief and concise. Provide your answers in 100 words or less."

first_message = "Hello, how can I help you today?"


def delete_chat_messages():
    for key in list(st.session_state.keys()):
        if key not in ["my_rag_text", "my_system_instructions", "my_llm_model", "my_space", "client",
                       "embeddings_model",
                       "my_sentences_rag", "my_sentences_rag_ids", "my_doc_ids", "my_sentences", "my_embeddings"]:
            del st.session_state[key]
    update_llm_model()


def create_sentences_rag():
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


# Create two columns with a 1:2 ratio
column_1, column_2 = st.columns([1, 2])

with column_1:
    st.expander("Disclaimer", expanded=False).markdown("""This application and code (hereafter referred to as the 'Software') is a proof of concept at an experimental stage and is not intended to be used as a production environment. The Software is provided as is, wihtout any warranties of any kind, expressed or implied and the user assumes full responsibility for its use, implementation, and legal compliance.

The developers of the Software shall not be liable for any damages, losses, claims, or liabilities arising from the Software, including but not limited to the usage of artificial intelligence and machine learning, related errors, third-party tool failures, security breaches, intellectual property violations, legal or regulatory non-compliance, deployment risks, or any indirect, incidental, or consequential damages.

Large Language Models may provide wrong answers. Please verify the answers and comply with applicable laws and regulations.

The user agrees to indemnify and hold harmless the developers of the Software from any related claims or disputes arising from the utilization of the Software by the user.

By using the Software, you agree to the terms and conditions of the disclaimer.""")

    model_list_all = ['mistralai/Mistral-7B-Instruct-v0.3',
                      'Qwen/Qwen2.5-72B-Instruct',
                      'HuggingFaceH4/zephyr-7b-beta']
    if os.getenv("GOOGLE_API_KEY"):
        model_list_all.append('gemini-2.5-flash-preview-05-20')
    if os.getenv("OPENAI_API_KEY"):
        model_list_all.append('gpt-4.1-nano-2025-04-14')
    st.selectbox("Select the model to use:",
                 model_list_all,
                 key="my_llm_model",
                 on_change=update_llm_model)

    st.text_area(label="Please enter your system instructions here:", value=my_system_instructions, height=80,
                 key="my_system_instructions", on_change=delete_chat_messages)

    rag_status_placeholder = st.empty()
    st.text_area(label="Please enter your RAG text here:", value=my_initial_rag_text, height=200, key="my_rag_text",
                 on_change=delete_chat_messages)

    st.slider("Minimum window size in original sentences", min_value=1, max_value=20, value=5, step=1,
              key="min_window_size", on_change=create_sentences_rag)

    st.slider("Maximum window size in original sentences", min_value=1, max_value=20, value=10, step=1,
              key="max_window_size", on_change=create_sentences_rag)

    st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="my_similarity_threshold")

    st.slider("Number of original chunks to keep", min_value=1, max_value=50, value=20, step=1,
              key="nof_keep_sentences")

    # Remove sub-prompt sliders since we're using full prompt now
    # st.slider("Minimum number of words in sub prompt split", min_value=1, max_value=10, value=1, step=1,
    #           key="nof_min_sub_prompts")
    #
    # st.slider("Maximum number of words in sub prompt split", min_value=1, max_value=10, value=5, step=1,
    #           key="nof_max_sub_prompts")

if "my_chat_messages" not in st.session_state:
    st.session_state['my_chat_messages'] = []
    st.session_state['my_chat_messages'].append(
        {"role": "system", "content": st.session_state['my_system_instructions']})

if "my_sentences_rag" not in st.session_state:
    create_sentences_rag()

with column_2:
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

            # Create bottom columns for displaying results
            bottom_col1, bottom_col2 = st.columns([1, 1])

            with bottom_col1:
                st.write(f"**Top matching chunks for:** '{prompt}'")
                st.write(f"Max similarity: {max_similarity:.4f}")

                if selected_chunk_indices:
                    for i, chunk_idx in enumerate(selected_chunk_indices[:8]):  # Show more chunks
                        doc_source = os.path.basename(st.session_state['my_doc_ids'][chunk_idx])
                        similarity_score = similarities[chunk_idx]
                        str_conf = (
                            f"Score: {similarity_score:.4f} "
                            f"(Source: {doc_source})"
                        )
                        with st.expander(f"Chunk {i + 1}: {str_conf}"):
                            chunk_text = st.session_state['my_sentences_rag'][chunk_idx]
                            st.write(f"**Chunk text:** {chunk_text[:200]}..." if len(
                                chunk_text) > 200 else f"**Chunk text:** {chunk_text}")
                            st.write(f"**Sentences used:** {len(st.session_state['my_sentences_rag_ids'][chunk_idx])}")
                else:
                    st.warning("No chunks found!")

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

        with bottom_col2:
            st.write("**Retrieval Information:**")
            retrieval_info = {
                "query": prompt,
                "max_similarity": float(max_similarity),
                "threshold": float(st.session_state['my_similarity_threshold']),
                "chunks_retrieved": len(selected_chunk_indices),
                "sentences_used": len(selected_sentence_indices),
                "total_chunks_available": len(st.session_state['my_sentences_rag'])
            }
            st.json(retrieval_info, expanded=True)

            st.write("**Recent Chat History:**")
            st.json(st.session_state['my_chat_messages'][-4:], expanded=False)