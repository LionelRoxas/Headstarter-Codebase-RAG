import os
from git import Repo
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
import glob

# Initialize clients
pinecone = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
pinecone_index = pinecone.Index("codebase-rag")

llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# Initialize the embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_file_content(file_path):
    """Read and return the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {
                "name": os.path.relpath(file_path),
                "content": f.read()
            }
    except Exception as e:
        st.error(f"Error reading file {file_path}: {str(e)}")
        return None

def process_repository(repo_path):
    """Process all code files in the repository."""
    # Define supported file extensions
    supported_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', '.go'}
    
    documents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if any(file.endswith(ext) for ext in supported_extensions):
                file_path = os.path.join(root, file)
                file_content = get_file_content(file_path)
                if file_content:
                    documents.append(file_content)
    return documents

def embed_and_store(documents):
    """Create embeddings and store in Pinecone."""
    model = get_embedding_model()
    
    for doc in documents:
        # Create embedding for the document
        embedding = model.encode(f"{doc['name']}\n{doc['content']}")
        
        # Store in Pinecone
        pinecone_index.upsert(
            vectors=[{
                "id": doc['name'],
                "values": embedding.tolist(),
                "metadata": {
                    "source": doc['name'],
                    "text": f"{doc['name']}\n{doc['content']}"
                }
            }],
            namespace="current-repo"
        )

def perform_rag(query):
    """Perform RAG to answer the query."""
    model = get_embedding_model()
    query_embedding = model.encode(query)
    
    # Search Pinecone
    results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace="current-repo"
    )
    
    # Construct prompt with context
    contexts = [item['metadata']['text'] for item in results['matches']]
    augmented_query = (
        "<CONTEXT>\n" + 
        "\n\n-------\n\n".join(contexts[:5]) + 
        "\n-------\n</CONTEXT>\n\n\nMY QUESTION:\n" + 
        query
    )
    
    # Get response from LLM
    system_prompt = """You are a Senior Software Engineer. Answer questions about the codebase based on the provided code context.
    Be specific and reference relevant code when appropriate."""
    
    response = llm_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    
    return response.choices[0].message.content

def clone_repository(repo_url):
    """Clones a GitHub repository."""
    try:
        repo_name = repo_url.split("/")[-1]
        base_path = os.path.join(os.getcwd(), "repositories")
        repo_path = os.path.join(base_path, repo_name)
        
        # Create directories and clean existing
        os.makedirs(base_path, exist_ok=True)
        if os.path.exists(repo_path):
            import shutil
            shutil.rmtree(repo_path)
        
        # Clone repository
        Repo.clone_from(repo_url, repo_path)
        
        # Process repository
        documents = process_repository(repo_path)
        embed_and_store(documents)
        
        return os.path.relpath(repo_path, os.getcwd())
    except Exception as e:
        st.error(f"Error processing repository: {str(e)}")
        return None

# Streamlit UI
st.title("Codebase RAG Assistant")

with st.form("repository_form"):
    url = st.text_input("Enter Github https URL:")
    submit = st.form_submit_button("Submit")

if submit and url:
    with st.spinner("Processing repository..."):
        path = clone_repository(url)
        if path:
            st.success('Repository successfully processed!', icon="✅")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new messages
if prompt := st.chat_input("Ask about the codebase..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and show response
    with st.spinner("Thinking..."):
        response = perform_rag(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
