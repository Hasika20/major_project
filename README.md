# 🏥 RAG-LLM Healthcare Insurance Assistant

## Description

An intelligent healthcare insurance document assistant powered by **FREE Local AI Models** and **Retrieval-Augmented Generation (RAG)**. This application allows you to upload complex healthcare insurance PDFs and get instant, accurate answers to your questions using state-of-the-art open-source AI - **completely free, no cloud costs!**

**Transform your healthcare insurance documents into an interactive knowledge base - at ZERO cost!** 🚀

### 💰 100% FREE - No Cloud Costs!

This project uses:

- ✅ **Sentence Transformers** - Free local embeddings (replaces AWS Titan)
- ✅ **Ollama + Llama 3.2** - Free local LLM (replaces AWS Nova)
- ✅ **Local Filesystem** - Free storage (replaces AWS S3)

**Monthly Cost: $0** (everything runs on your computer!)

## Demo Videos

### 🔍 Data Query Interface

![DataQuery](https://github.com/user-attachments/assets/198dc793-28e2-4541-9501-2ba00d27206d)

### 📚 Batch Processing Feature

![BatchProcessing_small](https://github.com/user-attachments/assets/3236d685-f939-4a7e-a6c0-92bfba17697c)

## ✨ Features

- **📄 Smart PDF Processing**: Upload healthcare insurance documents via intuitive Streamlit interface
- **🧠 FREE AI Embeddings**: Generate high-quality embeddings using Sentence Transformers (no cloud costs!)
- **💬 Intelligent Q&A**: Ask natural language questions and get contextual answers using Ollama (free local LLM)
- **💾 Local Storage**: Vector indexes saved to your computer (no S3 costs!)
- **🔒 Privacy First**: All data stays on your machine - never sent to cloud
- **⚡ No API Limits**: Unlimited queries, no rate limits, no usage caps
- **⚙️ Easy Configuration**: Simple one-time setup
- **🎯 Healthcare Focused**: Optimized for insurance terminology, policies, and procedures
- **💰 Zero Cost**: 100% free forever!

## 📋 Prerequisites

- **Python 3.8+**: Modern Python environment
- **Ollama**: Free local LLM runtime ([Download](https://ollama.com/download))
- **10GB Disk Space**: For AI models and vector stores
- **4-8GB RAM**: For running AI models locally

### 🆓 No Cloud Services Needed!

- ❌ No AWS account required
- ❌ No API keys needed
- ❌ No credit card required
- ✅ Everything runs on your computer!

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd RAG-LLM-Healthcare-Insurance

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r Admin/requirements.txt
```

### 2. Install Ollama (Free Local LLM)

**Windows:**

```bash
# Download from: https://ollama.com/download/windows
# Or use:
winget install Ollama.Ollama
```

**Verify installation:**

```bash
ollama --version
```

### 3. Download Free AI Model

```bash
# Recommended: Llama 3.2 (3B parameters, 2GB)
ollama pull llama3.2

# Test it works:
ollama run llama3.2
# Type "What is health insurance?" and press Ctrl+D to exit
```

### 4. Optional: Create .env file

Create a `.env` file (optional - not needed for free version):

```bash
# No AWS credentials needed!
# File is kept for backwards compatibility only
```

## 🎯 Usage

### Admin Interface - Document Processing

```bash
streamlit run Admin/admin.py --server.port 8501
```

**🌐 Open:** http://localhost:8501

**Features:**

- **📄 Single File Upload**: Upload individual PDF files for processing
- **📚 Bulk Processing**: Process all PDF files in `pdf-sources` folder automatically
- Automatic text extraction and chunking
- Generate embeddings using Titan V2
- Store vector indexes in S3 with unique naming
- Real-time progress tracking and detailed results

### User Interface - Interactive Q&A

```bash
streamlit run User/app.py --server.port 8502
```

**🌐 Open:** http://localhost:8502

**Features:**

- Ask natural language questions
- Get AI-powered answers from your documents
- Context-aware responses using Nova Lite
- Real-time document search

## 🧪 Testing Your Setup

### Complete Test Suite

Run all tests to verify your system is working correctly:

```bash
# Run all tests
./run_tests.sh
# or
python3 run_tests.py
```

### Individual Tests

Test specific components:

```bash
# AWS connectivity
python3 tests/test_s3_connection.py

# AI model access
python3 tests/test_bedrock_simple.py

# Complete system integration
python3 tests/test_complete_system.py
```

### Step-by-Step Testing

1. **Upload Documents**: Use sample PDFs in `pdf-sources/` folder
2. **Verify Processing**: Check for "Vector store created successfully" message
3. **Test Queries**: Ask questions like:
   - "What is a deductible?"
   - "What services are covered under preventive care?"
   - "How does copayment work?"
4. **Check S3**: Verify vector store files uploaded to S3

### 📚 Bulk Processing Feature

Process all PDF files at once using the admin interface:

1. **Open Admin Interface**: http://localhost:8501
2. **Navigate to "Bulk Process All PDFs" tab**
3. **Review the list** of PDF files in `pdf-sources/` folder
4. **Click "Start Bulk Processing"** to process all files automatically
5. **Monitor progress** with real-time updates and progress bar
6. **Review results** with detailed processing statistics

**Alternative: Command Line Demo**

```bash
python3 demo_bulk_processing.py
```

## 📁 Project Structure

```
RAG-LLM-Healthcare-Insurance/
├── Admin/                   # 🏗️ Modular admin architecture
│   ├── admin.py              # 📄 Main admin interface entry point
│   ├── config.py             # ⚙️ Configuration and AWS client management
│   ├── s3_operations.py      # ☁️ Amazon S3 file operations
│   ├── pdf_processor.py      # 📚 PDF processing and vector stores
│   ├── bulk_processor.py     # 🔄 Bulk processing coordination
│   ├── ui_components.py      # 🎨 Streamlit UI components
│   ├── compatibility.py      # 🔗 Backward compatibility layer
│   ├── requirements.txt      # 📦 Python dependencies
│   ├── Dockerfile           # 🐳 Container configuration
│   └── README.md            # 📖 Architecture documentation
├── User/
│   └── app.py               # 💬 Question-answering interface
├── tests/                   # 🧪 Test suite
│   ├── test_s3_connection.py     # AWS S3 connectivity test
│   ├── test_bedrock_simple.py    # Bedrock model access test
│   ├── test_embedding_regions.py # Regional embedding test
│   ├── test_nova_converse.py     # Nova Lite conversation test
│   ├── test_boto3.py            # Boto3 configuration test
│   ├── test_admin_embedding.py  # Admin interface test
│   ├── test_user_interface.py   # User interface test
│   ├── test_bulk_processing.py  # Bulk PDF processing test
│   ├── test_complete_system.py  # End-to-end system test
│   └── README.md                # Test documentation
├── pdf-sources/             # 📚 Sample healthcare insurance PDFs
├── run_tests.py            # 🏃 Test runner script
├── run_tests.sh            # 🐚 Shell script for tests
├── demo_bulk_processing.py # 🔄 Bulk processing demo script
├── main.py                 # 🚀 Application entry point
├── .env                    # 🔐 Environment configuration
├── .gitignore             # 🚫 Git ignore rules
└── README.md              # 📖 This file
```

## 🔧 Technical Architecture

### 🏗️ Modular Design (v2.0)

The application follows a clean, modular architecture with separation of concerns:

- **Configuration Layer** (`config.py`): Centralized AWS client management
- **Storage Layer** (`s3_operations.py`): S3 file operations and duplicate checking
- **Processing Layer** (`pdf_processor.py`): PDF text extraction and vector creation
- **Orchestration Layer** (`bulk_processor.py`): Bulk processing coordination
- **Presentation Layer** (`ui_components.py`): Streamlit UI components
- **Compatibility Layer** (`compatibility.py`): Backward compatibility support

### AI Models Used

- **Sentence Transformers (all-MiniLM-L6-v2)** - FREE local embeddings
  - Generates 384-dimensional embeddings
  - 80MB model size
  - Optimized for semantic search
- **Ollama + Llama 3.2** - FREE local LLM
  - 3B parameter model (2GB)
  - Fast inference on CPU
  - Natural language generation for Q&A
  - Alternatives: Mistral (7B), Phi-3 (3.8B)

### Key Technologies

- **Streamlit**: Interactive web interfaces
- **LangChain**: Document processing and RAG pipeline
- **FAISS**: Vector similarity search
- **Sentence Transformers**: FREE local embeddings
- **Ollama**: FREE local LLM runtime
- **Local Filesystem**: FREE storage (replaces S3)
- **Modular Python Architecture**: Clean separation of concerns

### 💰 Cost Comparison

| Feature        | AWS (Cloud)        | This Project (Local) |
| -------------- | ------------------ | -------------------- |
| Embeddings     | $0.0001/1K tokens  | **FREE**             |
| LLM Inference  | $0.0006/1K tokens  | **FREE**             |
| Storage        | $0.023/GB/month    | **FREE**             |
| API Limits     | Yes                | **No limits!**       |
| Privacy        | Data sent to cloud | **100% private**     |
| **Total Cost** | **$5-20/month**    | **$0/month**         |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**❌ "Ollama not found" Error**

- Solution: Restart terminal after Ollama installation
- Or manually add to PATH: `C:\Users\<username>\AppData\Local\Programs\Ollama`

**❌ "Model not found" Error**

- Solution: Download the model
- Run: `ollama pull llama3.2`
- Check installed models: `ollama list`

**❌ Slow Response Times**

- Solution: Use smaller model (llama3.2 instead of mistral)
- Close other applications to free up RAM
- First-time model load takes longer (subsequent runs faster)

**❌ "Failed to load embedding model"**

- Solution: Check internet connection (needed for first-time download only)
- The 80MB model will download automatically on first run
- After download, works offline!

### Get Help

- **Setup Guide**: See [FREE_SETUP_GUIDE.md](FREE_SETUP_GUIDE.md) for detailed instructions
- Run the test suite: `./run_tests.sh` or `python3 run_tests.py` (tests updated for free version)
- Check individual test outputs in the `tests/` directory

---

**Built with ❤️ for healthcare insurance professionals - Now 100% FREE!** 💰✨
