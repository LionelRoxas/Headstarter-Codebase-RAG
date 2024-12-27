import os
from git import Repo
import streamlit as st
from streamlit_chat import message

def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory.

    Args:
        repo_url: The URL of the GitHub repository.

    Returns:
        The path to the cloned repository.
    """
    repo_name = repo_url.split("/")[-1]  # Extract repository name from URL
    repo_name = repo_url.split("/")[-1]  # Extract repository name from URL
    repo_path = os.path.join(os.getcwd(), "repositories", repo_name)
    Repo.clone_from(repo_url, repo_path)
    relative_repo_path = os.path.relpath(repo_path, os.getcwd())
    return relative_repo_path

with st.form("my_form"):
    url = st.text_input("Enter Github http url...")
    submit = st.form_submit_button("Submit")

if submit:
    print("Url: ", url)
    if url:
        print("Calling function clone_repository...")
        path = clone_repository(url)
        print("Rep has been clone to: ", path)
        st.success('Repository successfully added!', icon="✅")

    else:
        print("Please, type in a github repository url.")
        st.error('Please, type in a github repository url.', icon="🚨")




uploaded_files = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=True
)

# for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.write("filename:", uploaded_file.name)
    # st.write(bytes_data)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
