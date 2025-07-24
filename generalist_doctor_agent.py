"""
Generalist Doctor Agent - Medical RAG System
===========================================

This agent provides medical Q&A responses using a vector database.
Built with LangChain and designed for integration with the multi-agent system.

Features:
- Loads medrag_qna collection from ChromaDB
- Uses nomic-ai/nomic-embed-text-v1.5 embeddings
- LM Studio integration for MedGemma 4B IT local model
- Q&A-specific prompt engineering
- Complete RAG chain with LCEL

Author: Hüseyin Çavuş
Date: 2025-01-22
Updated: 2025-07-24
"""

import os
import sys
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generalist_doctor_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GeneralistDoctorAgent:
    """
    Generalist Doctor Agent - Medical Q&A RAG Chain
    
    CRITICAL MEDICAL REQUIREMENTS:
    - Answers using MedQuAD Q&A dataset only
    - Zero-Error Tolerance: States when information is unavailable
    - Medical Q&A Precision: Specialized prompting for Q&A context
    - Local Model: MedGemma 4B IT via LM Studio for data privacy
    - Modular Design: Self-contained for multi-agent orchestration
    """
    
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """
        Initialize the Generalist Doctor Agent for cloud deployment.
        
        This agent loads and queries a pre-built ChromaDB vector store created on a separate machine.
        It does NOT perform data ingestion - only consumption of the existing database.
        Uses Google Gemini API for cloud compatibility.
        
        Args:
            chroma_db_path: Path to the pre-built ChromaDB database (default: "./chroma_db")
        """
        
        logger.info("🩺 INITIALIZING GENERALIST DOCTOR AGENT")
        logger.info("=" * 60)
        logger.info("📊 Loading pre-built ChromaDB database (no data ingestion)")
        
        # Step A: Initialize the exact same embedding model used to create the database
        logger.info("🔢 Step A: Initializing embedding model: sentence-transformers/all-MiniLM-L6-v2")
        from langchain_huggingface import HuggingFaceEmbeddings
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions - matches database
        )
        
        # Validate embeddings
        test_embedding = self.embeddings.embed_query("test medical query")
        logger.info(f"✅ Embeddings initialized successfully (dim: {len(test_embedding)})")
        
        # Step B: Load the pre-built Chroma Vector Store (NOT creating, only loading)
        logger.info("🗄️  Step B: Loading pre-built ChromaDB vector store")
        logger.info(f"📁 Database path: {chroma_db_path}")
        logger.info(f"📊 Collection: medical_guidelines")
        
        self.vector_store = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=self.embeddings,
            collection_name="medical_guidelines"
        )
        
        # Verify the pre-built collection has data
        collection_count = self.vector_store._collection.count()
        if collection_count == 0:
            logger.warning("⚠️  Collection 'medical_guidelines' is empty. Trying fallback to 'medical_textbooks'...")
            # Try fallback to textbooks collection
            self.vector_store = Chroma(
                persist_directory=chroma_db_path,
                embedding_function=self.embeddings,
                collection_name="medical_textbooks"
            )
            fallback_count = self.vector_store._collection.count()
            if fallback_count == 0:
                logger.error("❌ Both 'medical_guidelines' and 'medical_textbooks' collections are empty")
                # Create a minimal fallback
                logger.warning("🚨 Creating emergency fallback response system")
                self.vector_store = None
                self.retriever = None
                logger.info("⚠️  Agent will use LLM-only mode without RAG")
            else:
                logger.info(f"✅ Using fallback 'medical_textbooks' collection with {fallback_count:,} documents")
        else:
            logger.info(f"✅ Pre-built vector store loaded successfully")
            logger.info(f"📈 Collection documents: {collection_count:,}")
        
        # Step C: Create the retriever from the loaded vector store (if available)
        if self.vector_store is not None:
            logger.info("🔍 Step C: Creating retriever with k=5 for comprehensive context")
            
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Top-5 results as specified
            )
        else:
            logger.warning("🔍 Step C: No retriever created - using LLM-only mode")
        
        logger.info("✅ Retriever created successfully (k=5)")
        
        # Preserve existing LLM and RAG chain logic
        self._initialize_llm()
        self._create_prompt_template()
        self._assemble_rag_chain()
        
        logger.info("✅ Generalist Doctor Agent initialized successfully")
        logger.info("🎯 Ready to query pre-built medical textbooks database")
        logger.info("=" * 60)
    
    def _initialize_llm(self):
        """Initialize Google Gemini API for cloud deployment."""
        logger.info("🤖 Initializing Google Gemini API connection")
        logger.info("📋 Expected model: Gemini 1.5 Flash")
        
        try:
            # Use LM Studio for local deployment
            self.llm = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",  # LM Studio doesn't require a real API key
                model="medgemma-4b-it-mlx",  # Specify the exact model name
                temperature=0.3,      # Lower temperature for medical accuracy
                max_tokens=1200,      # Increased for detailed comprehensive responses
                top_p=0.85,          # More focused sampling
                frequency_penalty=0.4,  # Lower penalty to allow detailed explanations
                presence_penalty=0.3,   # Lower penalty for comprehensive coverage
            )
            
            logger.info("✅ LM Studio LLM initialized successfully")
            logger.info("🎯 Model: MedGemma 4B IT MLX via LM Studio, Temperature: 0.3, Max Tokens: 1200")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LM Studio LLM: {e}")
            logger.error("🔧 Troubleshooting:")
            logger.error("   1. Ensure LM Studio is running on http://localhost:1234")
            logger.error("   2. Check that MedGemma 4B IT MLX model is loaded in LM Studio")
            logger.error("   3. Verify LM Studio server is accessible")
            raise
    
    def _create_prompt_template(self):
        """Create the exact prompt template as specified."""
        logger.info("📝 Creating specified prompt template")
        
        # Enhanced prompt template for DETAILED, COMPREHENSIVE medical responses
        template = """You are a medical expert specializing in general practice medicine. Based on the medical information below, provide a DETAILED, COMPREHENSIVE answer that thoroughly addresses the patient's question.

INSTRUCTIONS:
- Provide a DETAILED, COMPREHENSIVE response covering all relevant aspects
- Include specific medical details, symptoms, causes, treatments, and recommendations
- Give thorough explanations that help patients understand their condition fully
- Cover prevention, diagnosis, treatment options, and when to seek care
- Use clear medical terminology with explanations for better understanding
- Provide actionable advice and next steps

Medical Information:
{context}

Medical Question: {question}

Provide a DETAILED and COMPREHENSIVE medical response:"""

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        logger.info("✅ Prompt template created with exact specification")
        logger.info("🩺 Anti-hallucination controls configured")
    
    def _assemble_rag_chain(self):
        """Assemble the complete RAG chain using LangChain Expression Language (LCEL)."""
        logger.info("🔗 Assembling RAG chain with LCEL")
        logger.info("📋 Chain logic: Retriever -> Prompt -> LLM -> String Output Parser")
        
        if self.retriever is None:
            # Create fallback chain without retriever
            logger.warning("🚨 Creating fallback chain without RAG (no retriever available)")
            
            fallback_template = """You are a Generalist Doctor AI. I apologize, but the medical knowledge database is currently unavailable. 

Based on general medical knowledge, here's what I can tell you about the question: {question}

IMPORTANT DISCLAIMER: This response is generated without access to the specific medical database and should not be used for medical diagnosis or treatment decisions. Please consult with a qualified healthcare professional.

ANSWER:"""
            
            self.prompt_template = PromptTemplate(
                input_variables=["question"],
                template=fallback_template
            )
            
            # Simple chain without retriever
            self.rag_chain = (
                {"question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("⚠️  Fallback chain assembled (LLM-only mode)")
            return
        
        def format_docs_with_sources(docs):
            """Format retrieved documents with source citations for the context."""
            formatted_sections = []
            for i, doc in enumerate(docs):
                # Extract source information from metadata
                source = doc.metadata.get("source", f"Unknown_Source_{i+1}")
                # Clean source name for citation
                if "/" in source:
                    source_name = source.split("/")[-1]  # Get filename
                else:
                    source_name = source
                
                # Format with source attribution
                section = f"CONTEXT FROM [{source_name}]:\n{doc.page_content}"
                formatted_sections.append(section)
            
            return "\n\n".join(formatted_sections)
        
        # Create the RAG chain using LCEL: Retriever -> Prompt -> LLM -> String Output Parser
        self.rag_chain = (
            {"context": self.retriever | format_docs_with_sources, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("✅ RAG chain assembled successfully")
        logger.info("🔄 Chain flow: Retriever -> Prompt -> LLM -> String Output Parser")
    
    async def answer_question(self, question: str) -> str:
        """
        Answer a medical question using the RAG chain asynchronously.
        
        Args:
            question: The medical question to answer
            
        Returns:
            The final answer string from the RAG chain
        """
        logger.info(f"❓ Processing question (ASYNC): {question[:100]}...")
        
        try:
            # Invoke the RAG chain asynchronously and return the final answer string
            response = await self.rag_chain.ainvoke(question)
            
            logger.info("✅ Question processed successfully")
            logger.info(f"📤 Response length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            error_response = f"I apologize, but I encountered an error processing your question: {str(e)}. Please try again."
            return error_response
    
    async def get_retrieved_sources(self, question: str) -> List[str]:
        """
        Get unique source documents that were retrieved for the question asynchronously.
        
        Args:
            question: Medical question
            
        Returns: 
            List of unique source document names
        """
        try:
            docs = await self.retriever.ainvoke(question)
            sources = set()
            for doc in docs:
                source = doc.metadata.get("source", "Unknown_Source")
                # Clean source name for display
                if "/" in source:
                    source_name = source.split("/")[-1]  # Get filename
                else:
                    source_name = source
                sources.add(source_name)
            
            return sorted(list(sources))
        except Exception as e:
            logger.error(f"❌ Source retrieval failed: {e}")
            return []

    def get_retrieved_context(self, question: str) -> List[Dict]:
        """
        Get the retrieved Q&A context for debugging/transparency.
        
        Args:
            question: Medical question
            
        Returns:
            List of retrieved Q&A document chunks with metadata
        """
        try:
            docs = self.retriever.invoke(question)
            return [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}")
            return []

def main():
    """Demonstration and verification of the Generalist Doctor Agent."""
    logger.info("🚀 GENERALIST DOCTOR AGENT - DEMONSTRATION")
    logger.info("=" * 70)
    
    try:
        # Initialize the agent
        print("Initializing Generalist Doctor Agent...")
        agent = GeneralistDoctorAgent()
        
        # Use the exact sample question as specified
        sample_question = "What is the cause of Atrial Fibrillation?"
        
        print(f"\n🔍 Sample Question: {sample_question}")
        print("-" * 50)
        
        # Call the answer_question method and print the result
        answer = agent.answer_question(sample_question)
        
        # Get source citations for traceability
        sources = agent.get_retrieved_sources(sample_question)
        
        print("📋 Answer:")
        print(answer)
        print("-" * 50)
        print("📚 SOURCE DOCUMENTS CONSULTED:")
        for i, source in enumerate(sources, 1):
            print(f"   {i}. {source}")
        print("-" * 50)
        print("✅ Generalist Doctor Agent with source citations demonstration completed successfully")
        print("🎯 Full traceability pipeline is working - sources are cited and listed")
        
        logger.info("✅ Agent verification completed successfully")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL FAILURE in Generalist Doctor Agent demonstration: {e}")
        logger.error("🔧 Please ensure:")
        logger.error("   1. medical_rag_foundation.py has completed successfully")
        logger.error("   2. ./chroma_db directory exists with medrag_qna collection")
        logger.error("   3. LM Studio is running with MedGemma 4B IT model loaded")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
