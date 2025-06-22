# RAG Document Q&A System Setup Guide

This guide will help you set up and run the RAG (Retrieval-Augmented Generation) system that allows users to upload documents and ask questions based on their content.

## 🔧 Prerequisites

Before getting started, make sure you have:

1. **Python 3.8 or higher** installed on your system
2. **OpenAI API Key** - Get it from [OpenAI Platform](https://platform.openai.com/api-keys)
3. **Pinecone API Key** - Get it from [Pinecone Console](https://app.pinecone.io/)
4. **Tesseract OCR** (for image text extraction) - [Installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)

## 📋 Installation Steps

### 1. Clone or Download the Project

Create a new directory for your project and save the provided files:
- `app.py` (main application)
- `requirements.txt` (dependencies)

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv rag-env

# Activate virtual environment
# On Windows:
rag-env\Scripts\activate
# On macOS/Linux:
source rag-env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

**Windows:**
- Download installer from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
- Install and add to PATH

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### 5. Set up Environment Variables

Create a `.env` file in your project directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

**Important:** Replace the placeholder values with your actual API keys.

## 🚀 Running the Application

1. Make sure your virtual environment is activated
2. Run the Streamlit application:

```bash
streamlit run app.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## 📖 How to Use

### Step 1: Upload Documents
1. Go to the "Upload Documents" tab
2. Click "Choose files to upload"
3. Select one or more documents (supported formats: PDF, DOCX, TXT, CSV, Excel, Images)
4. Click "Process Documents" to parse and store them in the vector database

### Step 2: Ask Questions
1. Switch to the "Ask Questions" tab
2. Type your question in the input field
3. Click "Get Answer" to receive an AI-generated response based on your documents

## 📁 Supported File Formats

- **PDF** (.pdf) - Text extraction from PDF documents
- **Word** (.docx) - Microsoft Word documents
- **Text** (.txt) - Plain text files
- **CSV** (.csv) - Comma-separated values
- **Excel** (.xlsx, .xls) - Microsoft Excel spreadsheets
- **Images** (.png, .jpg, .jpeg) - Text extraction using OCR

## ⚙️ Configuration Options

### Vector Database Settings

- **Index Name**: You can customize the Pinecone index name in the sidebar
- **Chunk Size**: Documents are split into 1000-character chunks with 200-character overlap
- **Similarity Threshold**: Retrieved contexts must have >0.7 similarity score

### Model Settings

- **Embedding Model**: OpenAI text-embedding-ada-002
- **Chat Model**: GPT-4o for answer generation
- **Temperature**: Set to 0.1 for consistent responses

## 🔍 Features

### Document Processing
- **Multi-format Support**: Handles various document types
- **Text Chunking**: Intelligently splits large documents
- **Metadata Preservation**: Maintains source information
- **Progress Tracking**: Visual feedback during processing

### Question Answering
- **Semantic Search**: Uses vector similarity for relevant context retrieval
- **Context-Aware Responses**: GPT-4o generates answers based on retrieved content
- **Source Attribution**: Maintains connection to original documents
- **Chat History**: Keeps track of recent questions and answers

### User Interface
- **Clean Design**: Intuitive Streamlit interface
- **Real-time Status**: Shows API key status and processing progress
- **Error Handling**: Graceful error messages and recovery
- **Responsive Layout**: Works on different screen sizes

## 🛠️ Troubleshooting

### Common Issues

**1. API Key Errors**
- Ensure your `.env` file is in the correct directory
- Verify API keys are valid and have sufficient credits
- Check for any extra spaces or quotes in the `.env` file

**2. Pinecone Connection Issues**
- Verify your Pinecone API key is correct
- Check if you're using the correct region (default: us-east-1)
- Ensure your Pinecone plan supports the required features

**3. Document Processing Errors**
- Check file formats are supported
- Ensure files aren't corrupted
- For images, verify Tesseract is properly installed

**4. OCR Issues**
- Install Tesseract OCR system-wide
- For Windows, add Tesseract to your PATH
- Test OCR with: `tesseract --version`

### Performance Tips

- **Large Documents**: Break very large files into smaller chunks before upload
- **Batch Processing**: Upload multiple related documents together
- **Index Management**: Use different index names for different document sets
- **Memory Usage**: Monitor RAM usage with many documents

## 🔒 Security Considerations

- **API Keys**: Never commit `.env` files to version control
- **Document Privacy**: Uploaded documents are processed through OpenAI's API
- **Data Retention**: Check OpenAI and Pinecone data retention policies
- **Local Processing**: Consider local alternatives for sensitive documents

## 📊 Scaling and Production

For production deployment:

1. **Environment Variables**: Use proper environment variable management
2. **Authentication**: Add user authentication if needed
3. **Rate Limiting**: Implement API rate limiting
4. **Monitoring**: Add logging and monitoring
5. **Caching**: Consider caching frequently asked questions
6. **Database**: Store metadata in a proper database for large-scale usage

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure API keys are valid and have sufficient credits
4. Check the Streamlit logs for detailed error messages

## 📝 License

This project is provided as-is for educational and development purposes. Please ensure compliance with OpenAI and Pinecone terms of service when using their APIs.
