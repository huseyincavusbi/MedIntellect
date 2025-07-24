# 🏥 MedIntellect - Privacy-First Medical AI System

**Local Multi-Agent Medical Consultation System with ChromaDB and LM Studio**

> ⚠️ **DISCLAIMER**: This is an educational/research project. Not intended for actual medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.

---

## 🌟 Key Features

- **🏠 100% Local Processing**: Complete privacy with LM Studio integration
- **🤖## 📁 Project Structure

```
MedIntellect/
├── streamlit_app.py              # Main web interface
├── medical_system.py             # Core orchestration system
├── generalist_doctor_agent.py    # General medical Q&A agent
├── guideline_specialist_agent.py # Medical guidelines specialist
├── research_analyst_agent.py     # Research and analysis agent
├── build_sample_database.py      # Database builder
├── setup_check.py                # System verification script
├── requirements.txt              # Python dependencies
├── .streamlit/config.toml        # Streamlit configuration
└── README.md                     # This file
```rchitecture**: Three specialized medical AI agents working in parallel
- **📚 Vector Database**: ChromaDB with 188K+ medical documents for RAG
- **⚡ Fast Response**: Local inference with no cloud dependencies
- **💰 Zero API Costs**: No external service fees
- **🔒 HIPAA-Friendly**: All data remains on your machine
- **� Interactive Web UI**: Professional Streamlit interface
- **🔄 Real-time Updates**: Live consultation tracking and follow-up support

## 🖥️ System Interface

![MedIntellect Medical Consultation Interface](attachments/medical_consultation_interface.png)
*Medical consultation interface showing user query about brain cancer symptoms*

![MedIntellect Detailed Medical Response](attachments/medical_response_detailed.png)
*Comprehensive medical response with detailed symptom breakdown and specialist information*

---

## 🏗️ System Architecture

### Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    MedIntellect System                          │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Web Interface (streamlit_app.py)                    │
├─────────────────────────────────────────────────────────────────┤
│  Medical System Orchestrator (medical_system.py)               │
│  ├── LangGraph Router                                          │
│  ├── Agent Coordinator                                         │
│  └── Response Synthesizer                                      │
├─────────────────────────────────────────────────────────────────┤
│               Three Specialized Medical Agents                  │
│  ┌───────────────┬──────────────────┬────────────────────────┐  │
│  │   Generalist  │    Guidelines    │    Research Analyst    │  │
│  │    Doctor     │   Specialist     │       Agent           │  │
│  │    Agent      │     Agent        │                       │  │
│  └───────────────┴──────────────────┴────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                ChromaDB Vector Database                         │
│  ├── medical_guidelines (188K+ documents)                      │
│  └── medical_textbooks                                         │
├─────────────────────────────────────────────────────────────────┤
│  LM Studio + MedGemma 4B IT MLX (localhost:1234)              │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### **🤖 Multi-Agent System**

1. **🩺 Generalist Doctor Agent** (`generalist_doctor_agent.py`)
   - **Purpose**: General medical Q&A, symptom analysis, health advice
   - **Database**: `medical_guidelines` collection (primary)
   - **Fallback**: `medical_textbooks` collection if guidelines unavailable
   - **Specialization**: Broad medical knowledge, patient-friendly explanations

2. **📋 Guidelines Specialist Agent** (`guideline_specialist_agent.py`)
   - **Purpose**: Clinical guidelines, protocols, evidence-based recommendations
   - **Database**: `medical_guidelines` collection (188K+ documents)
   - **Specialization**: Clinical decision support, treatment protocols
   - **Features**: Anti-hallucination prompts, medical precision focus

3. **� Research Analyst Agent** (`research_analyst_agent.py`)
   - **Purpose**: Medical research synthesis, latest studies, PubMed integration
   - **Database**: Multiple collections + PubMed API fallback
   - **Specialization**: Research literature analysis, evidence synthesis
   - **Features**: Hybrid RAG + API approach for comprehensive coverage

#### **📚 ChromaDB Vector Database**

**Database Structure:**
```
chroma_db_test/chroma_db/
├── medical_guidelines/     # 188K+ medical documents
└── medical_textbooks/      # Medical textbook content
```

**Technical Specifications:**
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Dimensions**: 384
- **Similarity Search**: Cosine similarity with top-k retrieval
- **Chunk Strategy**: 1000 tokens with 150 token overlap
- **Collections**: Specialized medical knowledge domains

#### **� RAG (Retrieval-Augmented Generation) Flow**

```
User Question → Embedding Generation → Vector Similarity Search
     ↓                    ↓                      ↓
ChromaDB Query → Document Retrieval → Context Assembly
     ↓                    ↓                      ↓
Prompt Template → LM Studio (MedGemma) → Medical Response
```

**Code Implementation:**
```python
# RAG Chain Assembly (LCEL Pattern)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
```

#### **🎯 LM Studio Integration**

**Model Configuration:**
- **Model**: MedGemma 4B IT MLX (Medical-specialized Gemma)
- **Endpoint**: `http://localhost:1234/v1` (OpenAI-compatible API)
- **Temperature**: 0.3 (balanced creativity/accuracy)
- **Max Tokens**: 1200 (detailed responses)
- **Context**: RAG-retrieved medical documents

**LLM Setup per Agent:**
```python
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="medgemma-4b-it-mlx",
    temperature=0.3,
    max_tokens=1200
)
```

#### **🌐 Web Interface**

**Streamlit Application** (`streamlit_app.py`):
- **Responsive Design**: Dark mode optimized
- **Real-time Processing**: Live consultation status
- **Session Management**: Conversation history and follow-ups
- **Error Handling**: Graceful fallbacks and user feedback
- **Medical Result Display**: Enhanced visibility with custom CSS

---

## 🚀 Quick Start Guide

### **Prerequisites**

1. **LM Studio**
   - Download from [LM Studio website](https://lmstudio.ai/)
   - Install and configure for your system

2. **MedGemma 4B IT MLX Model**
   - Load in LM Studio
   - Start local server on `localhost:1234`

3. **Python Environment**
   - Python 3.8+ required
   - Virtual environment recommended

### **Installation Steps**

#### **1. Clone & Setup**
```bash
# Clone the repository
git clone https://github.com/huseyincavusbi/MedIntellect.git
cd MedIntellect

# Create virtual environment
python -m venv medicalrag
source medicalrag/bin/activate  # On Windows: medicalrag\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### **2. Configure LM Studio**
```bash
# 1. Start LM Studio
# 2. Load MedGemma 4B IT MLX model
# 3. Start local server (localhost:1234)
# 4. Verify API endpoint is active
```

#### **3. Initialize Database**
```bash
# Option A: Use sample database (11 documents - quick testing)
python build_sample_database.py

# Option B: Use full database (188K+ documents - production)
# Ensure chroma_db_test/ directory exists with full database
```

#### **4. Launch Application**
```bash
# Start the web interface
streamlit run streamlit_app.py

# Access at: http://localhost:8501
```

#### **5. Verify Setup (Optional)**
```bash
# Run setup verification script
python setup_check.py

# This checks all dependencies and configurations
```

### **First-Time Setup Verification**

1. **✅ LM Studio Running**: Check `http://localhost:1234/v1/models`
2. **✅ Database Loaded**: Verify document counts in UI
3. **✅ Agents Initialized**: All three agents show ready status
4. **✅ Test Query**: Try "What are symptoms of diabetes?"

---

## 📁 Project Structure

```
MedicalRAG/
├── Core System Files
│   ├── streamlit_app.py              # Main web interface
│   ├── medical_system.py             # Multi-agent orchestrator
│   ├── generalist_doctor_agent.py    # General medical Q&A agent
│   ├── guideline_specialist_agent.py # Clinical guidelines agent
│   └── research_analyst_agent.py     # Medical research agent
│
├── Database & Setup
│   ├── build_sample_database.py      # Sample database creator
│   ├── setup_check.py                # System verification script
│   ├── chroma_db_test/               # Main vector database (188K+ docs)
│   └── DATABASE_SETUP.md             # Database configuration guide
│
├── Configuration
│   ├── requirements.txt              # Python dependencies
│   ├── .streamlit/config.toml        # Streamlit configuration
│   └── packages.txt                  # System packages
│
├── Documentation
│   ├── README.md                     # This file
│   └── DATABASE_SETUP.md             # Database setup guide
│
└── Development Tools
    ├── chromadb-data-ingestion.ipynb # Database ingestion notebook
    └── setup_check.py               # System verification script
```

---

## 🔧 Configuration Options

### **Database Paths**
The system auto-detects databases in priority order:
1. `./chroma_db_test/chroma_db` (188K+ documents - **recommended**)
2. `./chroma_db` (custom database location)
3. `./chroma_db_sample` (11 documents - testing only)

### **LM Studio Configuration**
```python
# Modify in agent files for custom settings
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "medgemma-4b-it-mlx"
TEMPERATURE = 0.3
MAX_TOKENS = 1200
```

### **Streamlit Customization**
```toml
# .streamlit/config.toml
[theme]
base = "dark"
primaryColor = "#22c55e"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
```

---

## 🎯 Usage Examples

### **Basic Medical Query**
```
User: "What are the symptoms of hypertension?"

System Flow:
1. Question → Embedding Generation
2. ChromaDB Vector Search → Retrieve relevant docs
3. Context Assembly → Prompt Template
4. LM Studio (MedGemma) → Generate response
5. Medical Response → Display in UI
```

### **Follow-up Questions**
```
User: "What lifestyle changes help with blood pressure?"

System:
- Maintains conversation context
- Leverages previous query for enhanced relevance
- Provides continuity across multiple questions
```

### **Multi-Agent Consultation**
```
Complex Query: "Latest research on diabetes treatment protocols"

Agent Coordination:
1. Guidelines Specialist → Clinical protocols
2. Research Analyst → Latest studies
3. Generalist Doctor → Patient-friendly explanation
4. Synthesis → Comprehensive response
```

---

## 🛠️ Troubleshooting

### **Common Issues**

#### **LM Studio Connection Error**
```
Error: Connection refused to localhost:1234
Solution:
1. Verify LM Studio is running
2. Check model is loaded
3. Ensure local server is started
4. Test endpoint: curl http://localhost:1234/v1/models
```

#### **Database Not Found**
```
Error: No ChromaDB database found
Solution:
1. Run: python build_sample_database.py
2. Or ensure chroma_db_test/ directory exists
3. Check database path in agent configurations
```

#### **Streamlit App Issues**
```
Error: Module not found
Solution:
1. Activate virtual environment
2. Reinstall: pip install -r requirements.txt
3. Check Python version (3.8+ required)
```

### **Performance Optimization**

#### **For Better Response Times**
- Ensure LM Studio uses GPU acceleration
- Use SSD storage for database files
- Allocate sufficient RAM (8GB+ recommended)

#### **For Better Accuracy**
- Use full database (188K+ documents) over sample
- Enable verbose logging for debugging
- Adjust temperature settings for use case

---

## 🔒 Privacy & Security

### **Data Privacy**
- **100% Local Processing**: No data leaves your machine
- **No Cloud Dependencies**: All inference happens locally
- **No Telemetry**: Anonymous usage tracking disabled
- **Secure Storage**: Database files remain on local filesystem

### **Medical Data Compliance**
- **HIPAA-Friendly**: Suitable for healthcare environments
- **No External APIs**: Patient data never transmitted
- **Audit Trail**: All queries logged locally
- **Access Control**: Single-user local deployment

---

## 🤝 Contributing

This is an educational project. For improvements:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/improvement`
3. **Make changes and test thoroughly**
4. **Submit pull request with detailed description**

### **Development Guidelines**
- Maintain medical accuracy and safety
- Follow existing code patterns
- Add comprehensive documentation
- Test with multiple medical scenarios

---

## � License

This project is for educational and research purposes. Please ensure compliance with local regulations when using medical AI systems.

---

## 🙏 Acknowledgments

- **HuggingFace**: Embedding models and transformers
- **ChromaDB**: Vector database technology
- **LangChain**: RAG framework and orchestration
- **LM Studio**: Local LLM inference platform
- **Streamlit**: Web application framework
- **Medical Community**: Open medical datasets and research

---

**⚡ Ready to start your private medical consultation system? Follow the Quick Start Guide above!**

### Local Processing Flow
```
Streamlit UI → Medical System → LM Studio (localhost:1234)
                     ↓
    [Generalist] [Guidelines] [Research] ← ChromaDB Vector Store
                     ↓
              Response Synthesis
```

## � Project Structure

```
MedicalRAG/
├── streamlit_app.py              # Main web interface
├── medical_system.py             # Core orchestration system
├── generalist_doctor_agent.py    # General medical Q&A agent
├── guideline_specialist_agent.py # Medical guidelines specialist
├── research_analyst_agent.py     # Research and analysis agent
├── build_sample_database.py      # Database builder
├── chroma_db_sample/            # Pre-built medical database (11 documents)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 💡 Usage Examples

### **General Medical Questions:**
- "What are the symptoms of diabetes?"
- "How is hypertension treated?"
- "What are the side effects of chemotherapy?"

### **Clinical Guidelines:**
- "What are the latest breast cancer screening guidelines?"
- "How should acute myocardial infarction be treated?"
- "What are the contraindications for ACE inhibitors?"

### **Research & Studies:**
- "What is the latest research on COVID-19 treatments?"
- "Are there new developments in Alzheimer's disease?"
- "What vaccines are recommended for adults?"

## � Configuration

### **LM Studio Settings:**
- **Model**: MedGemma 4B IT MLX
- **Port**: 1234 (default)
- **API**: OpenAI-compatible
- **Temperature**: 0.1 (for medical accuracy)
- **Max Tokens**: 600

### **Database Options:**
The system automatically detects and uses available databases:
1. `./chroma_db_sample` (included - 11 medical documents)
2. `./chroma_db_test` (if available)
3. `./chroma_db` (if available)

## �️ Troubleshooting

### **Common Issues:**

1. **"Connection error"**
   - Ensure LM Studio is running
   - Check model is loaded
   - Verify port 1234 is accessible

2. **"Empty collections"**
   - Run: `python build_sample_database.py`
   - Check `chroma_db_sample/` exists

3. **Import errors**
   - Run: `pip install -r requirements.txt`
   - Ensure Python 3.8+ is used

4. **Slow responses**
   - Check LM Studio model is fully loaded
   - Reduce temperature setting
   - Ensure adequate RAM available

## 🔐 Privacy & Security

- ✅ **100% Local Processing**: All data stays on your machine
- 🔒 **No Data Collection**: No information sent to external servers
- 🛡️ **HIPAA-Friendly**: Suitable for privacy-sensitive environments
- 💾 **Local Storage**: All databases and responses stored locally
- 🚫 **No Internet Required**: Works completely offline (after setup)

## 📄 License

This project is designed for research and educational use. Please ensure compliance with medical AI regulations in your jurisdiction.

## ⚠️ Medical Disclaimer

**IMPORTANT MEDICAL DISCLAIMER:**

This system is designed for educational and research purposes only. It is **NOT** intended to:
- Provide medical diagnosis
- Replace professional medical advice
- Serve as a substitute for consultation with qualified healthcare providers
- Make treatment recommendations

**Always consult with qualified healthcare professionals for medical advice, diagnosis, and treatment.**

---

**🏠 MedIntellect Local - Your Private Medical AI Assistant**  
*Powered by LM Studio, ChromaDB, and Local-First Architecture*
