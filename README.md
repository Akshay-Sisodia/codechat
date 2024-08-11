# Code-Aware Conversational Chatbot

A Streamlit app that enables users to interact with and analyze code files through a conversational interface. The app leverages language models to provide code insights, generate code snippets, and answer programming-related queries.

## Features

- **Upload Code Files**: Allows users to upload and analyze individual code files.
- **Local Repository Loading**: Indexes and loads code from a local repository.
- **Conversational Interface**: Engage in conversation to get code-related answers, analysis, or code generation.

## Getting Started

### Prerequisites

- Python 3.7 or later
- Required Python packages (listed in `requirements.txt`)
- Ollama CLI installed on your system

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Akshay-Sisodia/codechat.git
   cd codechat
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**:
   - Follow the installation instructions from the [Ollama website](https://ollama.com/docs/installation) to install the Ollama CLI.

5. **Pull Llama 3.1**:
   - Once Ollama is installed, you can pull the Llama 3.1 model by running:
   ```bash
   ollama pull llama3.1
   ```

6. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload Code Files**:
   - Use the file uploader in the sidebar to upload Python, JavaScript, Java, C++, Ruby, Go, or Rust files.
   - Files will be indexed and added to the knowledge base for analysis.

2. **Load Local Repository**:
   - Enter the path to a local code repository and click "Load Repository" to index all supported code files from the directory.

3. **Chat Interface**:
   - Use the chat interface to ask questions about the uploaded code or request code generation.
   - The chatbot can provide insights, analyze code, or generate code snippets based on your input.

## Example

1. **Upload a File**: Upload a Python file to add it to the system's knowledge base.
2. **Ask a Question**: Query the chatbot about specific functions or logic in the uploaded file.
3. **Request Code**: Describe your coding needs, and the chatbot will generate the corresponding code snippet.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

## Contact

For any questions or support, please reach out to [akshaysisodia.studies@gmail.com](mailto:akshaysisodia.studies@gmail.com).