# Educational Chat Assistant
This application enables students to chat with educational content from PDF documents.

## Features
- Create vector embeddings from PDF documents by standard, subject, and chapter
- Chat with specific chapters using AI-powered conversations
- Organize and manage your educational content
- Delete vectors when no longer needed

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Setup Instructions
1. Clone this repository or download the source files
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```
   pip install streamlit langchain langchain-google-genai langchain-community faiss-cpu
   ```
4. Set up your Google API key:
   ```
   export GOOGLE_API_KEY=your_api_key_here
   ```
   On Windows:
   ```
   set GOOGLE_API_KEY=your_api_key_here
   ```

## Usage
1. Run the setup script to create necessary directories:
   ```
   python setup.py
   ```
2. Start the application:
   ```
   streamlit run app.py
   ```
3. Open your browser and navigate to `http://localhost:8501`

## Application Structure
- **app.py**: Main Streamlit application with UI
- **vector_manager.py**: Handles PDF processing and vector operations
- **chat_engine.py**: Manages the chat functionality with LangChain
- **setup.py**: Initial setup script

## How to Use
1. **Create Vectors**: 
   - Go to the "Create Vectors" tab
   - Enter the standard, subject, and chapter
   - Upload a PDF file
   - Click "Process PDF"

2. **Chat with Content**:
   - Go to the "Chat" tab
   - Select the standard, subject, and chapter from the sidebar
   - Click "Chat with this chapter"
   - Ask questions about the content

3. **Manage Vectors**:
   - Go to the "Manage Vectors" tab
   - Select vectors to delete
   - Click "Delete Selected Chapters"

## Technologies Used
- Streamlit: UI framework
- LangChain: Framework for LLM applications
- Google Generative AI (Gemini): LLM model
- FAISS: Vector database for storing embeddings
