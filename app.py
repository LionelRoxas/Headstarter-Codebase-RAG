import os
from git import Repo
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import textwrap

# Constants
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java', 
                       '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

CHUNK_SIZE = 1000  # Size of each chunk in characters
METADATA_LIMIT = 40000  # Pinecone's metadata limit in bytes

# Initialize clients
pinecone = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
pinecone_index = pinecone.Index("codebase-rag")

llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks of approximately equal size."""
    return textwrap.wrap(text, chunk_size, break_long_words=False, break_on_hyphens=False)

def get_file_content(file_path, repo_path):
    """Get content of a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        st.error(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path: str):
    """Get content of supported code files from the repository."""
    files_content = []
    try:
        for root, _, files in os.walk(repo_path):
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)
    except Exception as e:
        st.error(f"Error reading repository: {str(e)}")
    return files_content

def process_repository(repo_path):
    """Process repository files and store in Pinecone."""
    files_content = get_main_files_content(repo_path)
    documents = []
    
    for file in files_content:
        content = file['content']
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            # Create a truncated metadata string
            metadata_content = f"{file['name']} (part {i+1}/{len(chunks)})\n{chunk[:500]}..."
            
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": file['name'],
                    "chunk": i,
                    "total_chunks": len(chunks),
                    "text": metadata_content
                }
            )
            documents.append(doc)
    
    # Batch process documents
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            vectorstore = PineconeVectorStore.from_documents(
                documents=batch,
                embedding=HuggingFaceEmbeddings(),
                index_name="codebase-rag",
                namespace=repo_path
            )
        except Exception as e:
            st.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
    
    return files_content

def perform_rag(query):
    """Execute RAG pipeline."""
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query_embedding = model.encode(query)
    
    results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True
    )
    
    contexts = []
    for match in results['matches']:
        if 'text' in match['metadata']:
            contexts.append(match['metadata']['text'])
    
    augmented_query = (
        "<CONTEXT>\n" + 
        "\n\n-------\n\n".join(contexts[:5]) + 
        "\n-------\n</CONTEXT>\n\n\nMY QUESTION:\n" + 
        query
    )
    
    response = llm_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a Senior Software Engineer. Answer questions about the codebase based on the provided context."},
            {"role": "user", "content": augmented_query}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Codebase RAG Assistant")

# Repository input form
with st.form("repository_form"):
    url = st.text_input("Enter Github https URL:")
    submit = st.form_submit_button("Submit")

if submit and url:
    with st.spinner("Processing repository..."):
        try:
            repo_path = os.path.join(os.getcwd(), "repositories", url.split("/")[-1])
            os.makedirs(os.path.dirname(repo_path), exist_ok=True)
            if os.path.exists(repo_path):
                import shutil
                shutil.rmtree(repo_path)
            
            Repo.clone_from(url, repo_path)
            files_content = process_repository(repo_path)
            st.success('Repository successfully processed!', icon="✅")
        except Exception as e:
            st.error(f"Error processing repository: {str(e)}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the codebase..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        response = perform_rag(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
