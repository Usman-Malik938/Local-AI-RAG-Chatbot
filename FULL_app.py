import streamlit as st
import os
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import asyncio
from io import BytesIO
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import docx
import requests
from PyPDF2 import PdfReader

# Path to the Ollama CLI
OLLAMA_PATH = '/usr/local/bin/ollama'

def run_ollama_query(prompt):
    """Run a query using the Ollama CLI."""
    try:
        result = subprocess.run(
            [OLLAMA_PATH, 'run', 'llama3.1'],
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        print("ollama output is : ",result)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error occurred: {e}"

# Loading environment variables from a .env file
load_dotenv()

# Function to extract text from uploaded files (PDF, DOCX, or others).
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text

def get_pdf_text(uploaded_file):
    text = ""
    pdf = BytesIO(uploaded_file.read())
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = [para.text for para in doc.paragraphs]
    return ' '.join(allText)

def get_csv_text(file):
    return "a"

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_vector_store(text_chunks, group_name):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embeddings_model.embed_documents(text_chunks)
    print("embeddings  : ", len(embeddings))
    vector_store = FAISS.from_texts(text_chunks, embeddings_model)
    vector_store.save_local(f"faiss_index_{group_name}")
    st.session_state[f'vector_store_{group_name}'] = vector_store
    return vector_store

def load_vector_store(group_name):
    if f'vector_store_{group_name}' in st.session_state:
        return st.session_state[f'vector_store_{group_name}']
    else:
        return FAISS.load_local(f"faiss_index_{group_name}")

def get_conversational_chain():
    model_name = "llama3.1"
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, then return "I don't have the information".\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    
    def generate_response(context, question):
        prompt = prompt_template.format(context=context, question=question)
        response = requests.post(
            'http://localhost:11400/v1/generate',
            json={"model": model_name, "prompt": prompt}
        )
        result = response.json()
        return result["response"]

    return generate_response

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "upload some docs and ask me a question"}]

def user_input(user_question, group_name):
    vector_store = load_vector_store(group_name)
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embeddings_model.embed_documents([user_question])[0]
    search_results = vector_store.similarity_search_by_vector(query_embedding)
    context = " ".join([doc.page_content for doc in search_results])
    prompt = f"""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, then answer according to your knowledge base.\n\n
    Context:\n{context}\n
    Question:\n{user_question}\n
    Answer:
    """
    print("Context is : ",context)
    response_text = run_ollama_query(prompt)
    return {"output_text": response_text}

def main():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    st.title("Chat with your files")

    with st.sidebar:
        # Group management
        group_name = st.text_input("Enter Group Name", key="group_name_input")
        if st.button("Create Group"):
            if group_name and group_name not in st.session_state.get('groups', []):
                st.session_state.groups = st.session_state.get('groups', []) + [group_name]
                st.success(f"Group '{group_name}' created successfully!")

        if 'groups' in st.session_state:
            selected_group = st.selectbox("Select a Group", st.session_state.groups)
        else:
            st.warning("No groups available. Create one first.")
            selected_group = None

        # File uploader and processing
        if selected_group:
            docs = st.file_uploader(
                f"Upload files for the '{selected_group}' group", type=["pdf", "docx", "txt"], accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_files_text(docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, selected_group)
                    st.success("Documents processed and indexed.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "upload some docs and ask me a question"}]

    if selected_group:
        st.write(f"Chatting with the '{selected_group}' group documents")
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = user_input(prompt, selected_group)
                        placeholder = st.empty()
                        full_response = ''
                        output_text = response['output_text']
                        for item in output_text:
                            full_response += item
                            placeholder.markdown(full_response)
                        placeholder.markdown(full_response)

if __name__ == "__main__":
    main()

