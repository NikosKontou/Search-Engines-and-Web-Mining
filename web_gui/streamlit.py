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
        "my_similarity_threshold", "nof_keep_sentences", "hyde_responses",
        "current_responses", "current_standard_response", "current_hyde_response"
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
st.session_state.setdefault("hyde_responses", [])
st.session_state.setdefault("current_responses", {})
st.session_state.setdefault("current_standard_response", "")
st.session_state.setdefault("current_hyde_response", "")


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
        return {'selected_chunk_indices': [], 'selected_sentence_indices': [], 'similarities': similarities,
                'max_similarity': 0.0}

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


def generate_hypothetical_answer(prompt):
    """Generate a hypothetical answer using the LLM"""
    hyde_system_instructions = "You are a helpful assistant that generates a hypothetical answer to the user's question. Be brief and concise. Provide your answer in 100 words or less."

    try:
        hyde_messages = [
            {"role": "system", "content": hyde_system_instructions},
            {"role": "user", "content": prompt}
        ]

        hypothetical_answer = ""
        for packet in st.session_state['client'].chat.completions.create(
                messages=hyde_messages,
                model=st.session_state['my_llm_model'],
                stream=True,
                max_tokens=512):
            choice = getattr(packet, "choices", None)
            if not choice:
                continue
            first_choice = choice[0]
            delta = getattr(first_choice, "delta", None)
            content_piece = getattr(delta, "content", None)
            if not content_piece:
                continue
            hypothetical_answer += content_piece

        return hypothetical_answer.strip()
    except Exception as e:
        return f"Error generating hypothetical answer: {str(e)}"


def generate_response_with_retrieved_docs(prompt, selected_sentence_indices, retrieval_info=None):
    """Generate response using retrieved documents"""
    if not selected_sentence_indices:
        return "No relevant information found to answer this question.", retrieval_info

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
            choice = getattr(packet, "choices", None)
            if not choice:
                continue
            first_choice = choice[0]
            delta = getattr(first_choice, "delta", None)
            content_piece = getattr(delta, "content", None)
            if not content_piece:
                continue
            response += content_piece
    except Exception as e:
        response = f"Error generating response: {str(e)}"

    return response, retrieval_info


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

# Display chat history
for message in st.session_state['my_chat_messages']:
    role = message.get('role')
    content = message.get('content', '')
    msg_type = message.get('type', 'normal')  # 'normal' or 'hyde'

    if role == "user":
        messages_container.chat_message("user", avatar=":material/psychology_alt:").markdown(content)
    elif role == "assistant":
        if msg_type == 'hyde':
            # Display HyDE response with different styling
            with messages_container.expander(f"🔍 **HyDE Response**", expanded=True):
                st.markdown(content)
        else:
            # Display normal RAG response
            messages_container.chat_message("ai", avatar=":material/robot_2:").markdown(f"**Standard RAG:** {content}")

if prompt := st.chat_input("you may ask here your questions"):
    messages_container.chat_message("user", avatar=":material/psychology_alt:").markdown(prompt)

    # Create two columns for responses
    col1, col2 = messages_container.columns(2)

    # Standard RAG Response (left column)
    with col1:
        with st.spinner("Generating Standard RAG response..."):
            st.markdown("### Standard RAG Response")

            similarity_results = find_most_similar_chunks(prompt)
            selected_chunk_indices = similarity_results['selected_chunk_indices']
            selected_sentence_indices = similarity_results['selected_sentence_indices']
            similarities = similarity_results['similarities']
            max_similarity = similarity_results['max_similarity']

            # Generate response if we have good match
            if max_similarity > float(st.session_state['my_similarity_threshold']) and selected_sentence_indices:
                standard_response, _ = generate_response_with_retrieved_docs(prompt, selected_sentence_indices)
                st.markdown(standard_response)

                # Store retrieval info for standard RAG
                retrieved_doc_paths = []
                for i in selected_chunk_indices:
                    if i < len(st.session_state.get('my_doc_ids', [])):
                        retrieved_doc_paths.append(st.session_state['my_doc_ids'][i])
                unique_doc_names = sorted(list({os.path.basename(p) for p in retrieved_doc_paths}))

                standard_retrieval_info = {
                    "query": prompt,
                    "max_similarity": float(max_similarity),
                    "threshold": float(st.session_state['my_similarity_threshold']),
                    "chunks_retrieved": len(selected_chunk_indices),
                    "sentences_used": len(selected_sentence_indices),
                    "total_chunks_available": len(st.session_state.get('my_sentences_rag', [])),
                    "similar_documents_found": unique_doc_names,
                    "method": "standard_rag"
                }
            else:
                standard_response = (
                    f"I don't have enough relevant information to answer this question. "
                    f"The best match has {100 * float(max_similarity):.2f}% similarity, which is below the threshold of "
                    f"{100 * float(st.session_state['my_similarity_threshold']):.2f}%."
                )
                st.markdown(standard_response)
                standard_retrieval_info = {
                    "method": "standard_rag",
                    "max_similarity": float(max_similarity),
                    "threshold": float(st.session_state['my_similarity_threshold']),
                    "status": "below_threshold"
                }

            # Store the standard response in session state
            st.session_state["current_standard_response"] = standard_response

    # HyDE Response (right column)
    with col2:
        with st.spinner("Generating HyDE response..."):
            st.markdown("### HyDE Response")

            # Step 1: Generate hypothetical answer
            hypothetical_answer = generate_hypothetical_answer(prompt)

            # Step 2: Use hypothetical answer to find similar documents
            hyde_similarity_results = find_most_similar_chunks(hypothetical_answer)
            hyde_selected_sentence_indices = hyde_similarity_results['selected_sentence_indices']
            hyde_max_similarity = hyde_similarity_results['max_similarity']
            hyde_selected_chunk_indices = hyde_similarity_results['selected_chunk_indices']

            # Step 3: Generate final answer using retrieved documents
            if hyde_max_similarity > float(
                    st.session_state['my_similarity_threshold']) and hyde_selected_sentence_indices:
                hyde_response, _ = generate_response_with_retrieved_docs(prompt, hyde_selected_sentence_indices)
                st.markdown(hyde_response)

                # Store retrieval info for HyDE
                hyde_retrieved_doc_paths = []
                for i in hyde_selected_chunk_indices:
                    if i < len(st.session_state.get('my_doc_ids', [])):
                        hyde_retrieved_doc_paths.append(st.session_state['my_doc_ids'][i])
                hyde_unique_doc_names = sorted(list({os.path.basename(p) for p in hyde_retrieved_doc_paths}))

                hyde_retrieval_info = {
                    "query": prompt,
                    "hypothetical_answer": hypothetical_answer[:500] + "..." if len(
                        hypothetical_answer) > 500 else hypothetical_answer,
                    "max_similarity": float(hyde_max_similarity),
                    "threshold": float(st.session_state['my_similarity_threshold']),
                    "chunks_retrieved": len(hyde_selected_chunk_indices),
                    "sentences_used": len(hyde_selected_sentence_indices),
                    "similar_documents_found": hyde_unique_doc_names,
                    "method": "hyde"
                }
            else:
                hyde_response = (
                    f"HyDE couldn't find relevant information. "
                    f"The hypothetical answer had {100 * float(hyde_max_similarity):.2f}% similarity, which is below the threshold of "
                    f"{100 * float(st.session_state['my_similarity_threshold']):.2f}%."
                )
                st.markdown(hyde_response)
                hyde_retrieval_info = {
                    "method": "hyde",
                    "hypothetical_answer": hypothetical_answer[:500] + "..." if len(
                        hypothetical_answer) > 500 else hypothetical_answer,
                    "max_similarity": float(hyde_max_similarity),
                    "threshold": float(st.session_state['my_similarity_threshold']),
                    "status": "below_threshold"
                }

            # Store the HyDE response in session state
            st.session_state["current_hyde_response"] = hyde_response

    # Store both responses for comparison
    st.session_state["current_responses"] = {
        "question": prompt,
        "standard": standard_response,
        "hyde": hyde_response,
        "hypothetical_answer": hypothetical_answer
    }

    # Store conversation
    st.session_state['my_chat_messages'].append({"role": "user", "content": prompt})
    st.session_state['my_chat_messages'].append({"role": "assistant", "content": standard_response, "type": "normal"})
    st.session_state['my_chat_messages'].append({"role": "assistant", "content": hyde_response, "type": "hyde"})

    # Store HyDE responses separately for comparison
    hyde_entry = {
        "question": prompt,
        "standard_response": standard_response,
        "hyde_response": hyde_response,
        "hypothetical_answer": hypothetical_answer,
        "timestamp": st.session_state.get("timestamp", 0)
    }
    st.session_state["hyde_responses"].append(hyde_entry)

    # Limit chat history
    if len(st.session_state['my_chat_messages']) > 20:
        st.session_state['my_chat_messages'] = [st.session_state['my_chat_messages'][0]] + st.session_state[
            'my_chat_messages'][-18:]

    # Display retrieval info in expanders
    with messages_container.expander("Retrieval Information", expanded=False):
        tab1, tab2 = st.tabs(["Standard RAG", "HyDE"])

        with tab1:
            st.write("**Standard RAG Retrieval Info:**")
            st.json(standard_retrieval_info, expanded=False)

        with tab2:
            st.write("**HyDE Retrieval Info:**")
            st.json(hyde_retrieval_info, expanded=False)

            # Show hypothetical answer details
            with st.expander("View Hypothetical Answer"):
                st.markdown(f"**Generated Hypothetical Answer:**")
                st.markdown(hypothetical_answer)

    # Add comparison section - This must be AFTER both responses are generated and stored
    with messages_container.expander("Compare Responses", expanded=False):
        # Get the current responses from session state
        current_responses = st.session_state.get("current_responses", {})

        if current_responses:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Standard RAG:**")
                # Use a container with border for better visibility
                with st.container(border=True):
                    if "standard" in current_responses:
                        st.markdown(current_responses["standard"])
                    else:
                        st.info("Standard response not available")
            with col2:
                st.markdown("**HyDE Response:**")
                with st.container(border=True):
                    if "hyde" in current_responses:
                        st.markdown(current_responses["hyde"])
                    else:
                        st.info("HyDE response not available")
        else:
            st.info("No responses available for comparison yet.")

# Add sidebar with HyDE settings and info
with st.sidebar:
    st.header("HyDE Settings")

    if st.session_state.get("hyde_responses"):
        st.subheader("Recent HyDE Responses")
        for i, entry in enumerate(reversed(st.session_state["hyde_responses"][-5:])):
            with st.expander(
                    f"Q: {entry['question'][:50]}..." if len(entry['question']) > 50 else f"Q: {entry['question']}"):
                st.markdown("**Question:**")
                st.write(entry['question'])
                st.markdown("**Hypothetical Answer:**")
                st.write(
                    entry['hypothetical_answer'][:200] + "..." if len(entry['hypothetical_answer']) > 200 else entry[
                        'hypothetical_answer'])
                st.markdown("**Final HyDE Response:**")
                st.write(entry['hyde_response'])

    if st.button("Clear HyDE History"):
        st.session_state["hyde_responses"] = []
        st.rerun()