import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.callbacks.base import BaseCallbackHandler
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters.html import HtmlFormatter
import time
import logging

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def read_code_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_language_from_extension(extension):
    language_mapping = {
        'py': Language.PYTHON,
        'js': Language.JS,
        'java': Language.JAVA,
        'cpp': Language.CPP,
        'rb': Language.RUBY,
        'go': Language.GO,
        'rs': Language.RUST
    }
    return language_mapping.get(extension.lower(), Language.PYTHON)  # Default to Python

def add_code_to_vectorstore(code, file_name, extension):
    language = get_language_from_extension(extension)
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=100,
        chunk_overlap=20
    )
    chunks = splitter.split_text(code)
    metadatas = [{"source": file_name} for _ in chunks]
    st.session_state.vector_store.add_texts(chunks, metadatas=metadatas)

def highlight_code(code, language):
    lexer = get_lexer_by_name(language, stripall=True)
    formatter = HtmlFormatter(style="monokai", noclasses=True)
    return highlight(code, lexer, formatter)

# Initialize Streamlit session state
if 'conversation' not in st.session_state:
    # Initialize Ollama
    llm = Ollama(model="phi3")
    embeddings = OllamaEmbeddings(model="phi3")

    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chroma_db_path = "./chroma_db"
    if os.path.exists(chroma_db_path):
        try:
            shutil.rmtree(chroma_db_path)
        except PermissionError as e:
            logging.error(f"Error deleting directory {chroma_db_path}: {e}")
            time.sleep(1)  # Wait before retrying
            try:
                shutil.rmtree(chroma_db_path)
            except Exception as ex:
                logging.error(f"Retry failed: {ex}")

    # Initialize vector store
    st.session_state.vector_store = Chroma(embedding_function=embeddings, persist_directory=chroma_db_path)

    # Initialize conversation chain
    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vector_store.as_retriever(),
        memory=memory
    )
    st.session_state.uploaded_files = {}

# Streamlit UI
st.title("Code-Aware Conversational Chatbot")

# Sidebar for file upload and repository selection
with st.sidebar:
    st.header("Code Management")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a code file", type=["py", "js", "java", "cpp", "rb", "go", "rs"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        code_content = read_code_file(tmp_file_path)
        uploaded_file_extension = uploaded_file.name.split('.')[-1]
        add_code_to_vectorstore(code_content, uploaded_file.name, uploaded_file_extension)
        st.session_state.uploaded_files[uploaded_file.name] = code_content
        st.success(f"File {uploaded_file.name} uploaded and added to the knowledge base.")
        os.unlink(tmp_file_path)
    
    # Local repository selection
    repo_path = st.text_input("Enter the path to your local repository:")
    if repo_path and os.path.isdir(repo_path):
        if st.button("Load Repository"):
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(('.py', '.js', '.java', '.cpp', '.rb', '.go', '.rs')):
                        file_path = os.path.join(root, file)
                        code_content = read_code_file(file_path)
                        file_extension = file.split('.')[-1]
                        add_code_to_vectorstore(code_content, file, file_extension)
            st.success("Repository loaded successfully!")

# Main chat interface
st.header("Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            language = "python"  # Default to Python for code snippets
            if message.get("language"):
                language = message["language"]
            st.markdown(highlight_code(message["content"], language), unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("You can discuss code, ask questions, or request code generation here."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        # Check if the user is asking about a specific file
        file_name = None
        for name in st.session_state.uploaded_files.keys():
            if name.lower() in prompt.lower():
                file_name = name
                break
        
        if file_name:
            file_content = st.session_state.uploaded_files[file_name]
            context = f"The user is asking about the file '{file_name}'. Here's the content of the file:\n\n{file_content}\n\nPlease analyze and respond to the user's query about this file."
            response = st.session_state.conversation({"question": context + "\n\n" + prompt}, callbacks=[stream_handler])
        else:
            code_prompt = """
            You are an expert programmer. Your task is to generate high-quality, efficient, and well-documented code based on the user's requirements. 
            Please follow these guidelines:
            1. Use best practices and design patterns appropriate for the language and task.
            2. Include clear and concise comments to explain complex logic.
            3. Handle potential errors and edge cases.
            4. Optimize for performance where possible. 
            5. Adhere to the language's style guide and common conventions.
            6. If applicable, suggest unit tests for the generated code.

            Now, please generate the code based on the following request:
            """
            code_request = code_prompt + prompt
            response = st.session_state.conversation({"question": code_request}, callbacks=[stream_handler])
        
        st.session_state.messages.append({"role": "assistant", "content": response['answer'], "language": "python"})  # Default to Python for code responses

