# Gyantra: NCERT Solutions

A Streamlit-based educational application that provides interactive chatbot solutions for NCERT (National Council of Educational Research and Training) learning materials.

## Overview

Gyantra NCERT Solutions is an AI-powered educational tool that helps students access context-aware answers to their questions about NCERT textbooks. The application uses RAG (Retrieval-Augmented Generation) to provide accurate responses based on the content of uploaded textbooks.

## Features

- **Chapter-based Conversations**: Chat with AI about specific chapters from NCERT textbooks
- **PDF Processing**: Upload PDF files of NCERT chapters for vectorization
- **Vector Management**: Create, view, and delete chapter vectors
- **Contextual Chat**: Get answers based on the selected chapter's content
- **Conversation History**: View and continue conversations within each session

## Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: FAISS for efficient similarity search
- **Embedding Model**: Google's Gemini Embedding model
- **LLM**: Google's Gemini 2.5 Flash model 
- **Document Processing**: LangChain for PDF handling and text splitting

## Setup Instructions

### Prerequisites

- Python 3.8+
- Google API key with access to Gemini models

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/azharbihari/gyantra.git
   cd gyantra
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

### Directory Structure

```
gyantra/
├── app.py                 # Main Streamlit application
├── services/
│   ├── chat_service.py    # Handles RAG and conversation
│   └── vector_service.py  # Manages vector creation and deletion
├── vectors/               # Storage for vector databases
└── requirements.txt       # Project dependencies
```

## Usage Guide

### Running the Application

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501` by default.

### Creating Chapter Vectors

1. Navigate to the "Manage Vectors" tab
2. Fill in the form with the standard (grade), subject, and chapter name
3. Upload the chapter's PDF file
4. Click "Process PDF" to create the vector database

### Chatting with Chapters

1. Go to the "Chat" tab
2. Select a chapter from the dropdown menu
3. Type your question in the chat input
4. Receive contextually relevant answers based on the chapter content

## Dependencies

The following main packages are required:
- streamlit
- langchain
- langchain-google-genai
- faiss-cpu
- pypdf

See requirements.txt for complete dependencies.

## Limitations

- Requires Google API key with access to Gemini models
- Currently supports only PDF files
- Vector databases can consume significant disk space for large textbooks

## Future Enhancements

- Support for DOCX and other file formats
- Multi-language support for regional language textbooks
- User authentication and persistence of chat history
- Advanced search and filtering options
- Exportable summaries and study notes

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- NCERT for educational materials
- Google Gemini for AI capabilities
- Streamlit for the interactive web interface
- LangChain for RAG framework