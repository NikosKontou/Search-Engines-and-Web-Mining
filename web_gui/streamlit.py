import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from huggingface_hub import InferenceClient
import os
import numpy as np
from openai import OpenAI

st.set_page_config(layout="wide")




# new code
# --- Initialize essential session state keys ---
for key, default_value in {
    'my_embeddings': np.array([]),
    'my_sentences': [],
    'my_sentences_rag_ids': [],
    'my_doc_ids': [],
    'my_chat_messages': [{"role": "system", "content": "You are a helpful assistant."}],
    'rag_docs': {},
    'min_window_size': 2,
    'max_window_size': 3,
    'nof_min_sub_prompts': 1,
    'nof_max_sub_prompts': 3,
    'nof_keep_sentences': 5,
    'my_similarity_threshold': 0.3,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value





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







# new code ends







# with open("./labeled_text_small.txt", "r", encoding="utf-8") as f:
#     my_initial_rag_text = f.read()

# Check if the LLM model is not already in the session state
if "my_llm_model" not in st.session_state:
    # Set the default LLM model to "mistralai/Mistral-7B-Instruct-v0.3"
    st.session_state['my_llm_model'] = "mistralai/Mistral-7B-Instruct-v0.3"

# Check if the SPACE_ID environment variable is not already in the session state
if "my_space" not in st.session_state:
    st.session_state['my_space'] = os.environ.get("SPACE_ID")


# Function to update the LLM model client
def update_llm_model():
    if st.session_state['my_llm_model'].startswith("gemini-"):
        # Initialize the client for gemini models. We use the OpenAI API to interact with gemini models.
        st.session_state['client'] = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"),
                                            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    elif st.session_state['my_llm_model'].startswith("gpt-"):
        # Initialize the client for openai models
        st.session_state['client'] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # ,base_url = "https://eu.api.openai.com/" # gives error
    else:
        if st.session_state['my_space']:
            # Initialize the client with the model if SPACE_ID is available
            st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'])
        else:
            # Initialize the client with the model and token if SPACE_ID is not available
            st.session_state['client'] = InferenceClient(st.session_state['my_llm_model'], token=os.getenv("HF_TOKEN"))


# Check if the client is not already in the session state
if "client" not in st.session_state:
    update_llm_model()

# Check if the embeddings model is not already in the session state
if "embeddings_model" not in st.session_state:
    # We will use the all-MiniLM-L6-v2 model for embeddings
    st.session_state['embeddings_model'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

my_system_instructions = "You are a helpful assistant that answers questions about the manga and tv-series one piece. Be brief and concise. Provide your answers in 100 words or less."

first_message = "Hello, how can I help you today?"


def delete_chat_messages():
    for key in st.session_state.keys():
        if key != "my_rag_text" and key != "my_system_instructions":
            del st.session_state[key]
    update_llm_model()


def create_sentences_rag():
    """
    Build rolling-window sentence chunks and embeddings for all RAG documents.
    Automatically rebuilds when text or window settings change.
    """
    if 'rag_docs' not in st.session_state or not st.session_state['rag_docs']:
        st.warning("No RAG documents loaded.")
        return

    pattern = r'(?<=[.!?])\s+'

    # Cache hash to detect content changes
    doc_hash = hash(str(st.session_state['rag_docs'])
                    + str(st.session_state['min_window_size'])
                    + str(st.session_state['max_window_size']))

    if st.session_state.get('last_doc_hash') == doc_hash:
        st.info("RAG documents unchanged — using cached embeddings.")
        return

    # Initialize/reset storage
    st.session_state['my_sentences'] = []
    st.session_state['my_sentences_rag'] = []
    st.session_state['my_sentences_rag_ids'] = []
    st.session_state['my_doc_ids'] = []
    all_embeddings = []
    all_sentences = []

    st.info("Building RAG embeddings…")

    for doc_path, doc_text in st.session_state['rag_docs'].items():
        # ✅ split into sentences safely
        sentences = [s.strip() for s in re.split(pattern, doc_text) if s.strip()]

        if not sentences:
            st.warning(f"No sentences found in {doc_path}. Skipping.")
            continue

        doc_chunks = []
        doc_chunk_ids = []

        for rolling_window_size in range(
            st.session_state['min_window_size'],
            st.session_state['max_window_size'] + 1
        ):
            # ✅ Prevent negative range bug
            if len(sentences) < rolling_window_size:
                continue

            for i in range(0, len(sentences) - rolling_window_size + 1):
                chunk = " ".join(sentences[i:i + rolling_window_size]).strip()
                if chunk:
                    doc_chunks.append(chunk)
                    doc_chunk_ids.append(list(range(i, i + rolling_window_size)))
                    st.session_state['my_doc_ids'].append(doc_path)

        if doc_chunks:
            try:
                embeddings = st.session_state['embeddings_model'].encode(doc_chunks)
                all_embeddings.extend(embeddings)
                all_sentences.extend(sentences)
                st.session_state['my_sentences_rag'].extend(doc_chunks)
                st.session_state['my_sentences_rag_ids'].extend(doc_chunk_ids)
            except Exception as e:
                st.error(f"Embedding failed for {doc_path}: {e}")
                continue

    if all_embeddings:
        st.session_state['my_embeddings'] = np.array(all_embeddings)
        st.session_state['my_sentences'] = all_sentences
        st.session_state['last_doc_hash'] = doc_hash
        st.success(f"✅ Indexed {len(all_sentences)} sentences across {len(st.session_state['rag_docs'])} documents.")
    else:
        st.warning("No embeddings created — check your document contents.")



# Create two columns with a 1:2 ratio
column_1, column_2 = st.columns([1, 2])

# In the first column
with column_1:
    # Display a disclaimer about the potential inaccuracies of Large Language Models
    st.expander("Disclaimer", expanded=False).markdown("""This application and code (hereafter referred to as the 'Software') is a proof of concept at an experimental stage and is not intended to be used as a production environment. The Software is provided as is, wihtout any warranties of any kind, expressed or implied and the user assumes full responsibility for its use, implementation, and legal compliance.

The developers of the Software shall not be liable for any damages, losses, claims, or liabilities arising from the Software, including but not limited to the usage of artificial intelligence and machine learning, related errors, third-party tool failures, security breaches, intellectual property violations, legal or regulatory non-compliance, deployment risks, or any indirect, incidental, or consequential damages.

Large Language Models may provide wrong answers. Please verify the answers and comply with applicable laws and regulations.

The user agrees to indemnify and hold harmless the developers of the Software from any related claims or disputes arising from the utilization of the Software by the user.

By using the Software, you agree to the terms and conditions of the disclaimer.""")

    # Add a selectbox for model selection
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

    # Add a text are for the system instructions
    st.text_area(label="Please enter your system instructions here:", value=my_system_instructions, height=80,
                 key="my_system_instructions", on_change=delete_chat_messages)

    # Placeholder right after text_area
    rag_status_placeholder = st.empty()
    # Add a text area for RAG text input
    st.text_area(label="Please enter your RAG text here:", value=my_initial_rag_text, height=200, key="my_rag_text",
                 on_change=delete_chat_messages)

    # Add a slider for minimum window size
    st.slider("Minimum window size in original sentences", min_value=1, max_value=20, value=5, step=1,
              key="min_window_size", on_change=create_sentences_rag)

    # Add a slider for maximum window size
    st.slider("Maximum window size in original sentences", min_value=1, max_value=20, value=10, step=1,
              key="max_window_size", on_change=create_sentences_rag)

    # Add a slider for the similarity threshold
    st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="my_similarity_threshold")

    # Add a slider for the number of sentences to keep
    st.slider("Number of original chunks to keep", min_value=1, max_value=50, value=20, step=1,
              key="nof_keep_sentences")

    # Add a slider for the number of minimum sub prompts
    st.slider("Minimum number of words in sub prompt split", min_value=1, max_value=10, value=1, step=1,
              key="nof_min_sub_prompts")

    # Add a slider for the number of maximum sub prompts
    st.slider("Maximum number of words in sub prompt split", min_value=1, max_value=10, value=5, step=1,
              key="nof_max_sub_prompts")

# Check if the chat messages are not already in the session state
if "my_chat_messages" not in st.session_state:
    # Initialize the chat messages list in the session state
    st.session_state['my_chat_messages'] = []
    # Add the system instructions to the chat messages
    st.session_state['my_chat_messages'].append(
        {"role": "system", "content": st.session_state['my_system_instructions']})

# print(100*"-")
# Check if the sentences are not already in the session state
if "my_sentences_rag" not in st.session_state:
    create_sentences_rag()

with column_2:
    # Create a container for the messages with a specified height
    messages_container = st.container(height=500)

    # Display the first message from the assistant
    messages_container.chat_message("ai", avatar=":material/robot_2:").markdown(first_message)

    # Iterate through the chat messages stored in the session state
    for message in st.session_state['my_chat_messages']:
        if message['role'] == "user":
            # Display user messages with a specific avatar - https://fonts.google.com/icons
            messages_container.chat_message(message['role'], avatar=":material/psychology_alt:").markdown(
                message['content'])
        elif message['role'] == "assistant":
            # Display assistant messages with a specific avatar
            messages_container.chat_message(message['role'], avatar=":material/robot_2:").markdown(message['content'])

    # Check if there is a new prompt from the user
    # Check if there is a new prompt from the user
    if prompt := st.chat_input("you may ask here your questions"):

        # 🧹 Reset per-question context (avoid reuse)
        st.session_state.pop('similarities_to_question', None)
        st.session_state.pop('current_context_indices', None)

        # 🧩 Split the prompt into sub-prompts for rolling comparisons
        split_prompt = prompt.split()
        all_sub_prompts = []
        for jj in range(st.session_state['nof_min_sub_prompts'],
                        st.session_state['nof_max_sub_prompts'] + 1):
            for ii in range(len(split_prompt)):
                sub = " ".join(split_prompt[ii:ii + jj]).strip()
                if sub:
                    all_sub_prompts.append(sub)

        # 🧠 Compute similarity scores fresh for this question
        similarities_to_question = np.zeros(len(st.session_state['my_embeddings']))
        # 🧠 Compute similarity scores fresh for this question
        embeddings = st.session_state.get('my_embeddings', np.array([]))
        if embeddings.size == 0:
            st.warning("No embeddings available — please load or reindex documents before asking questions.")
            st.stop()

        similarities_to_question = np.zeros(len(embeddings))
        for sub_prompt in all_sub_prompts:
            q_emb = st.session_state['embeddings_model'].encode([sub_prompt])
            similarities_to_question += cosine_similarity(q_emb, embeddings).flatten()

        for sub_prompt in all_sub_prompts:
            q_emb = st.session_state['embeddings_model'].encode([sub_prompt])
            similarities_to_question += cosine_similarity(
                q_emb, st.session_state['my_embeddings']
            ).flatten()

        similarities_to_question /= max(len(all_sub_prompts), 1)
        st.session_state['similarities_to_question'] = similarities_to_question

        # 🧮 Rank by similarity
        sorted_indices_rag = similarities_to_question.argsort()[::-1]
        sorted_indices_sentences = []
        max_similarity = 0.0

        bottom_col1, bottom_col2 = st.columns([1, 1])

        irag = 0
        while len(set(sorted_indices_sentences)) < st.session_state['nof_keep_sentences'] and irag < len(
                sorted_indices_rag):
            sorted_indices_sentences.extend(st.session_state['my_sentences_rag_ids'][sorted_indices_rag[irag]])
            max_similarity = max(max_similarity, similarities_to_question[sorted_indices_rag[irag]])

            with bottom_col1:
                doc_source = os.path.basename(st.session_state['my_doc_ids'][sorted_indices_rag[irag]])
                str_conf = (
                    f"Confidence: {similarities_to_question[sorted_indices_rag[irag]]:.5f} "
                    f"(Source: {doc_source})"
                )
                with st.expander(f"Chunk {irag + 1}: {str_conf}"):
                    for idx in st.session_state['my_sentences_rag_ids'][sorted_indices_rag[irag]]:
                        st.write(st.session_state['my_sentences'][idx])
            irag += 1

        sorted_indices_sentences = sorted(list(set(sorted_indices_sentences)))
        st.session_state['current_context_indices'] = sorted_indices_sentences

        # 💬 Display user question
        messages_container.chat_message("user", avatar=":material/psychology_alt:").markdown(prompt)

        # 🧠 Prepare RAG context dynamically
        with messages_container.chat_message("ai", avatar=":material/robot_2:"):
            response_placeholder = st.empty()
            response = ""
            augmented_prompt = ""

            if max_similarity > st.session_state['my_similarity_threshold']:
                context_text = "\n".join(
                    [st.session_state['my_sentences'][idx] for idx in sorted_indices_sentences]
                )

                augmented_prompt = (
                        "This is my context:\n\n" + "-" * 20 + "\n\n" +
                        context_text +
                        "\n\n" + "-" * 20 + "\n\n" +
                        "If the above context is not relevant, ignore it. "
                        "If it is relevant, answer based on both the context and the prompt.\n\n"
                        + "-" * 20 + "\n\n" +
                        f"The prompt is:\n\n{prompt}"
                )

                # Use a local copy of the chat history (no session mutation)
                conversation = st.session_state['my_chat_messages'] + [
                    {"role": "user", "content": augmented_prompt}
                ]

                for chunk in st.session_state['client'].chat.completions.create(
                        messages=conversation,
                        model=st.session_state['my_llm_model'],
                        stream=True,
                        max_tokens=1024,
                ):
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
                        response_placeholder.markdown(response)
            else:
                response = (
                    f"I do not have enough information to reply. "
                    f"The maximum similarity found in the context is: {100 * max_similarity:.2f}%."
                )
                response_placeholder.markdown(response)

        # 💾 Store message history (safe & capped)
        st.session_state['my_chat_messages'].append({"role": "user", "content": prompt})
        st.session_state['my_chat_messages'].append({"role": "assistant", "content": response})

        if len(st.session_state['my_chat_messages']) > 10:
            st.session_state['my_chat_messages'] = (
                    st.session_state['my_chat_messages'][:1] +
                    st.session_state['my_chat_messages'][3:]
            )

        with bottom_col2:
            st.write("Augmented prompt:")
            st.json({"max_similarity": max_similarity, "augmented_prompt": augmented_prompt}, expanded=False)
            st.write("Messages History All:")
            st.json(st.session_state['my_chat_messages'], expanded=False)
