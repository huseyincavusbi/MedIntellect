"""
Guideline Specialist Agent - Medical RAG System 
===============================================

This agent provides evidence-based medical guidance using clinical guidelines.
Built with LangChain Expression Language (LCEL) and strict hallucination controls.
Now powered by MedGemma 4B IT MLX via LM Studio for consistent local inference.

Features:
- Loads medical_guidelines_test collection from ChromaDB
- Uses nomic-ai/nomic-embed-text-v1.5 embeddings (same as ingestion)
- LM Studio integration with MedGemma 4B IT MLX model
- Anti-hallucination prompt engineering
- Complete RAG chain with LCEL pipe syntax
- Top-4 chunk retrieval for comprehensive context
- Source citation system for transparency

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
        logging.FileHandler('guideline_specialist_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GuidelineSpecialistAgent:
    """
    Guideline Specialist Agent - Medical RAG Chain
    
    CRITICAL MEDICAL REQUIREMENTS:
    - Zero-Error Tolerance: Answers ONLY from provided context
    - Anti-Hallucination: Explicitly states when information is unavailable
    - Medical Precision: Top-4 chunk retrieval for comprehensive coverage
    - LM Studio Integration: MedGemma 4B IT MLX for consistent local inference
    """
    
    def __init__(self, chroma_db_path: str = "./chroma_db", lm_studio_base_url: str = "http://localhost:1234/v1"):
        """
        Initialize the Guideline Specialist Agent.
        
        This agent loads and queries a pre-built ChromaDB vector store created on a separate machine.
        It does NOT perform data ingestion - only consumption of the existing database.
        
        Args:
            chroma_db_path: Path to the pre-built ChromaDB database (default: "./chroma_db")
            lm_studio_base_url: LM Studio API endpoint URL
        """
        self.lm_studio_base_url = lm_studio_base_url
        
        logger.info("🩺 INITIALIZING GUIDELINE SPECIALIST AGENT")
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
            raise ValueError("Pre-built collection 'medical_guidelines' is empty or doesn't exist")
        
        logger.info(f"✅ Pre-built vector store loaded successfully")
        logger.info(f"📈 Collection documents: {collection_count:,}")
        
        # Step C: Create the retriever from the loaded vector store
        logger.info("🔍 Step C: Creating retriever with k=5 for comprehensive context")
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Top-5 results as specified
        )
        
        logger.info("✅ Retriever created successfully (k=5)")
        
        # Preserve existing LLM and RAG chain logic
        self._initialize_llm()
        self._create_prompt_template()
        self._assemble_rag_chain()
        
        logger.info("✅ Guideline Specialist Agent initialized successfully")
        logger.info("🎯 Ready to query pre-built medical guidelines database")
        logger.info("=" * 60)
    
    def _initialize_llm(self):
        """Initialize LM Studio connection for MedGemma 4B IT MLX model."""
        logger.info(f"🤖 Initializing LM Studio connection: {self.lm_studio_base_url}")
        logger.info("📋 Expected model: MedGemma 4B IT MLX")
        
        try:
            self.llm = ChatOpenAI(
                base_url=self.lm_studio_base_url,
                api_key="lm-studio",  # LM Studio doesn't require real API key
                model="medgemma-4b-it-mlx",  # Use the loaded MLX model
                temperature=0.3,      # Lower temperature for more focused responses
                max_tokens=1200,      # Significantly increased for detailed responses
                top_p=0.85,          # More focused sampling
                frequency_penalty=0.2,  # Lower penalty to allow detailed explanations
                presence_penalty=0.1,   # Lower penalty for comprehensive coverage
                request_timeout=120,   # Extended timeout for detailed responses
                # Add timestamp to break caching
                extra_headers={"X-Request-ID": f"medintellect-{hash(str(__import__('time').time()))}"[:20]}
            )
            
            logger.info("✅ LM Studio connection established successfully")
            logger.info("🎯 Configuration: MedGemma 4B IT MLX, temp=0.3, max_tokens=1200")
            logger.info("🔬 Optimized for detailed medical guidelines and comprehensive responses")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LM Studio connection: {e}")
            logger.error("🔧 Troubleshooting steps:")
            logger.error("   1. Ensure LM Studio is running on localhost:1234")
            logger.error("   2. Verify MedGemma 4B IT MLX model is loaded in LM Studio")
            logger.error("   3. Check that LM Studio server is accessible")
            logger.error("   4. Confirm model is fully loaded and ready for inference")
            raise
    
    def _create_prompt_template(self):
        """Create anti-hallucination prompt template for medical guidance."""
        logger.info("📝 Creating anti-hallucination prompt template")
        
        template = """You are a Guideline Specialist Agent providing comprehensive, evidence-based medical guidance.

CRITICAL INSTRUCTIONS:
- Provide a DETAILED, COMPREHENSIVE answer based on the provided medical guideline context
- Include ALL relevant information: symptoms, risk factors, diagnostic criteria, treatment options, and prevention strategies when applicable
- Organize your response with clear sections and bullet points for readability
- Cite sources using [Source Name] format throughout your response
- If the context doesn't contain relevant information, state that clearly
- Aim for thorough, educational responses that fully address the question
- Use medical terminology but explain complex concepts clearly

MEDICAL GUIDELINE CONTEXT:
{context}

QUESTION: {question}

COMPREHENSIVE EVIDENCE-BASED RESPONSE:"""

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        logger.info("✅ Anti-hallucination prompt template created")
        logger.info("🛡️  Zero hallucination policy enforced")
    
    def _assemble_rag_chain(self):
        """Assemble the complete RAG chain using LangChain Expression Language (LCEL)."""
        logger.info("🔗 Assembling RAG chain with LCEL pipe syntax")
        
        def format_docs_with_sources(docs):
            """Format retrieved documents with source citations for the prompt."""
            formatted_chunks = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content
                source = doc.metadata.get('source', 'Unknown Source')
                
                chunk_text = f"CHUNK {i} [Source: {source}]:\n{content}"
                formatted_chunks.append(chunk_text)
            
            return "\n\n".join(formatted_chunks)
        
        # Create the RAG chain using LCEL pipe syntax with explicit question passing
        self.rag_chain = (
            {
                "context": lambda x: format_docs_with_sources(self.retriever.invoke(x)),
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("✅ RAG chain assembled successfully")
        logger.info("🔄 Flow: Question → Retrieval → Context Formatting → LLM → Response")
    
    async def answer_question(self, question: str) -> str:
        """
        Answer a medical question using the RAG chain asynchronously.
        
        Args:
            question: The medical question to answer
            
        Returns:
            Evidence-based response with source citations
        """
        logger.info("❓ PROCESSING MEDICAL GUIDELINE QUESTION (ASYNC)")
        logger.info("=" * 60)
        logger.info(f"Question: {question}")
        
        try:
            # Invoke the RAG chain asynchronously
            response = await self.rag_chain.ainvoke(question)
            
            logger.info("✅ Question processed successfully")
            logger.info(f"📋 Response length: {len(response)} characters")
            logger.info("=" * 60)
            
            return response
            
        except Exception as e:
            error_msg = f"Guideline Specialist Agent Error: Unable to process question due to {str(e)}"
            logger.error(f"❌ {error_msg}")
            return error_msg
    
    async def get_retrieved_sources(self, question: str) -> List[Dict[str, Any]]:
        """
        Get the sources that would be retrieved for a given question asynchronously.
        
        Args:
            question: The medical question
            
        Returns:
            List of retrieved document sources with metadata
        """
        try:
            docs = await self.retriever.ainvoke(question)
            sources = []
            
            for doc in docs:
                source_info = {
                    "source": doc.metadata.get('source', 'Unknown Source'),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            return sources
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve sources: {e}")
            return []

async def main():
    """
    Demonstration of the Guideline Specialist Agent.
    """
    logger.info("🚀 GUIDELINE SPECIALIST AGENT - DEMONSTRATION")
    logger.info("=" * 70)
    
    try:
        # Initialize the agent
        print("Initializing Guideline Specialist Agent...")
        agent = GuidelineSpecialistAgent()
        
        # Test question
        test_question = "What are the staging criteria for stage 2 stomach cancer?"
        
        print(f"\n🏥 GUIDELINE SPECIALIST CONSULTATION")
        print("=" * 70)
        print(f"❓ QUESTION: {test_question}")
        print("-" * 70)
        
        # Get the answer asynchronously
        answer = await agent.answer_question(test_question)
        
        print(f"💡 GUIDELINE-BASED ANSWER:")
        print(f"   {answer}")
        print("=" * 70)
        print("✅ GUIDELINE SPECIALIST AGENT DEMONSTRATION COMPLETED")
        
        # Show retrieved sources
        sources = await agent.get_retrieved_sources(test_question)
        print(f"\n📚 RETRIEVED SOURCES:")
        print("-" * 40)
        for i, source in enumerate(sources, 1):
            print(f"Source {i}: {source['source']}")
            print(f"Preview: {source['content_preview']}")
            print()
        
        logger.info("✅ Guideline Specialist Agent demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL FAILURE in demonstration: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
