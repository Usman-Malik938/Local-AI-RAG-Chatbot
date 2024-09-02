import os
import subprocess
from fastapi import FastAPI, Form, UploadFile
from io import BytesIO
import requests
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import docx
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Path to the Ollama CLI executable
OLLAMA_PATH = '/usr/local/bin/ollama'

def run_ollama_query(prompt):
    """Run a query using the Ollama CLI."""
    try:
        # Execute the Ollama CLI command with the given prompt and capture the output
        result = subprocess.run(
            [OLLAMA_PATH, 'run', 'llama3.1'],
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        # Return the output from the command
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Return an error message if the command fails
        return f"Error occurred: {e}"

# Initialize the SentenceTransformer model for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_vector_store(text_chunks):
    """Create a FAISS vector store from text chunks."""
    # Initialize the HuggingFaceEmbeddings model with the SentenceTransformer model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a FAISS vector store from the text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embeddings_model)
    
    # Save the FAISS vector store locally
    vector_store.save_local("faiss_index")
    
    # Return the vector store object
    return vector_store

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    """Handle question queries and return the response."""
    # Load the previously saved FAISS vector store with the embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)

    # Generate an embedding for the user's question
    query_embedding = embeddings_model.embed_documents([question])[0]
    
    # Perform a similarity search in the vector store using the query embedding
    search_results = vector_store.similarity_search_by_vector(query_embedding)
    
    # Extract the content of the most similar documents to form the context
    context = " ".join([doc.page_content for doc in search_results])
    
    # Construct the prompt for the Ollama CLI with the context and question
    prompt = f"""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, then answer according to your knowledge base.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    # Run the Ollama query with the constructed prompt
    response_text = run_ollama_query(prompt)
    
    # Return the response from the Ollama query
    return {"response": response_text}

@app.post("/upload/")
async def upload_file(uploaded_files: list[UploadFile]):
    """Handle file uploads and process them into a FAISS vector store."""
    raw_text = ""
    try:
        for uploaded_file in uploaded_files:
            # Get the file extension to determine the file type
            file_extension = os.path.splitext(uploaded_file.filename)[1]
            
            # Read the content of the uploaded file
            file_content = await uploaded_file.read()  # Await the read coroutine
            
            # Process the file based on its extension
            if file_extension == ".txt":
                # Decode text files and add to raw_text
                raw_text += file_content.decode('utf-8')
            elif file_extension == ".pdf":
                # Read and extract text from PDF files
                pdf = BytesIO(file_content)
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()
            elif file_extension == ".docx":
                # Read and extract text from DOCX files
                doc = docx.Document(BytesIO(file_content))
                for para in doc.paragraphs:
                    raw_text += para.text
            else:
                # Return an error if the file type is unsupported
                return {"error": "Unsupported file type"}
        
        # Split the raw text into smaller chunks for better processing
        text_splitter = CharacterTextSplitter(
            separator=" ",  # Use space as the separator
            chunk_size=50,  # Define the maximum size of each chunk
            chunk_overlap=10  # Define the overlap between chunks
        )
        text_chunks = text_splitter.split_text(raw_text)
        
        # Create and save the FAISS vector store using the text chunks
        get_vector_store(text_chunks)
        
        # Return a success message after processing the files
        return {"message": "Files processed and vector store created successfully."}
    
    except Exception as e:
        # Return an error message if any exception occurs during processing
        return {"error": str(e)}

# Run the FastAPI app using Uvicorn as the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=5000)  # Run the app on port 5000
