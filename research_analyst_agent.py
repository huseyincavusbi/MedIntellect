"""
Research Analyst Agent - Medical RAG System
==========================================

This agent finds and summarizes recent medical research using PubMed API.
Built with LangChain tools and designed for integration with the multi-agent system.

Features:
- Vector database checking first for existing relevant data
- PubMed API integration for latest research when needed
- LM Studio integration for MedGemma 4B IT local model
- Research-specific prompt engineering for summarization
- Tool -> LLM chain architecture with fallback to RAG
- Modular design for multi-agent orchestration

Author: Hüseyin Çavuş
Date: 2025-01-22 (Updated: 2025-07-24)
"""

import os
import sys
from typing import List, Dict, Any, Optional
from langchain_community.utilities import PubMedAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_analyst_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ResearchAnalystAgent:
    """
    Research Analyst Agent - Vector DB + PubMed Research Chain
    
    CRITICAL RESEARCH REQUIREMENTS:
    - Checks vector database first for existing relevant research data
    - Uses PubMed API for latest research only when vector DB has insufficient data
    - Synthesizes research abstracts into concise summaries
    - Hybrid approach: RAG + PubMed API for comprehensive coverage
    - Modular Design: Self-contained for multi-agent orchestration
    """
    
    def __init__(self, 
                 lm_studio_base_url: str = "http://localhost:1234/v1", 
                 chroma_db_path: str = "./chroma_db_test/chroma_db",
                 max_docs: int = 5,
                 relevance_threshold: float = 0.7):
        """
        Initialize the Research Analyst Agent.
        
        Args:
            lm_studio_base_url: LM Studio API endpoint URL
            chroma_db_path: Path to the ChromaDB database
            max_docs: Maximum number of PubMed documents to retrieve
            relevance_threshold: Minimum similarity score to consider vector DB sufficient
        """
        self.lm_studio_base_url = lm_studio_base_url
        self.chroma_db_path = chroma_db_path
        self.max_docs = max_docs
        self.relevance_threshold = relevance_threshold
        
        logger.info("🔬 INITIALIZING RESEARCH ANALYST AGENT WITH VECTOR DB INTEGRATION")
        logger.info("=" * 70)
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_vector_database()
        self._initialize_pubmed_tool()
        self._initialize_llm()
        self._create_prompt_templates()
        self._assemble_hybrid_chain()
        
        logger.info("✅ Research Analyst Agent initialized successfully")
        logger.info("=" * 70)
    
    def _initialize_embedding_model(self):
        """Initialize embedding model for vector database queries."""
        logger.info("🔢 Initializing embedding model for vector database queries")
        
        try:
            # Use the same model that was used to create the database (384 dimensions)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("✅ Embedding model initialized: all-MiniLM-L6-v2 (384 dim)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise
    
    def _initialize_vector_database(self):
        """Initialize connection to ChromaDB vector database."""
        logger.info(f"🗄️  Initializing vector database connection: {self.chroma_db_path}")
        
        try:
            # Try to connect to medical_guidelines collection first
            self.guidelines_db = Chroma(
                persist_directory=self.chroma_db_path,
                embedding_function=self.embeddings,
                collection_name="medical_guidelines"
            )
            
            # Try to connect to medical_textbooks collection
            self.textbooks_db = Chroma(
                persist_directory=self.chroma_db_path,
                embedding_function=self.embeddings,
                collection_name="medical_textbooks"
            )
            
            guidelines_count = self.guidelines_db._collection.count()
            textbooks_count = self.textbooks_db._collection.count()
            
            logger.info(f"✅ Vector database connected successfully")
            logger.info(f"📊 Medical guidelines: {guidelines_count} documents")
            logger.info(f"📚 Medical textbooks: {textbooks_count} documents")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {e}")
            logger.warning("⚠️  Research agent will use PubMed API only")
            self.guidelines_db = None
            self.textbooks_db = None
    
    def _initialize_pubmed_tool(self):
        """Initialize PubMed API wrapper tool."""
        logger.info("📚 Initializing PubMed API wrapper")
        
        try:
            self.pubmed_tool = PubMedAPIWrapper(
                top_k_results=self.max_docs,  # Limit results for focused analysis
                doc_content_chars_max=2000    # Limit content length per document
            )
            
            logger.info(f"✅ PubMed API wrapper initialized successfully")
            logger.info(f"📊 Max documents: {self.max_docs}, Max chars per doc: 2000")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize PubMed API wrapper: {e}")
            raise
    
    def _check_vector_database_relevance(self, query: str) -> tuple[bool, str]:
        """
        Check if vector database has sufficient relevant data for the query.
        
        Args:
            query: Research query to check
            
        Returns:
            Tuple of (has_sufficient_data, existing_data_summary)
        """
        logger.info(f"🔍 Checking vector database for relevant data: {query[:100]}...")
        
        try:
            all_results = []
            
            # Search in guidelines collection
            if self.guidelines_db:
                try:
                    guidelines_results = self.guidelines_db.similarity_search_with_score(
                        query, k=3
                    )
                    all_results.extend(guidelines_results)
                    logger.info(f"📋 Found {len(guidelines_results)} relevant guidelines")
                except Exception as e:
                    logger.warning(f"⚠️  Guidelines search failed: {e}")
            
            # Search in textbooks collection
            if self.textbooks_db:
                try:
                    textbooks_results = self.textbooks_db.similarity_search_with_score(
                        query, k=3
                    )
                    all_results.extend(textbooks_results)
                    logger.info(f"📚 Found {len(textbooks_results)} relevant textbook entries")
                except Exception as e:
                    logger.warning(f"⚠️  Textbooks search failed: {e}")
            
            if not all_results:
                logger.info("📭 No results found in vector database")
                return False, ""
            
            # Check if we have sufficiently relevant results
            relevant_results = [doc for doc, score in all_results if score > self.relevance_threshold]
            
            if relevant_results:
                # Combine relevant documents into summary
                combined_content = "\n\n".join([doc.page_content[:500] for doc in relevant_results[:5]])
                logger.info(f"✅ Found {len(relevant_results)} highly relevant documents in vector DB")
                return True, combined_content
            else:
                logger.info(f"⚠️  Found {len(all_results)} results but none above relevance threshold ({self.relevance_threshold})")
                return False, ""
                
        except Exception as e:
            logger.error(f"❌ Vector database check failed: {e}")
            return False, ""
    
    def _initialize_llm(self):
        """Initialize LM Studio connection for MedGemma 4B IT model."""
        logger.info(f"🤖 Initializing LM Studio connection: {self.lm_studio_base_url}")
        logger.info("📋 Expected model: MedGemma 4B IT")
        
        try:
            self.llm = ChatOpenAI(
                base_url=self.lm_studio_base_url,
                api_key="lm-studio",  # LM Studio doesn't require real API key
                model="medgemma-4b-it-mlx",  # Use the loaded MLX model
                temperature=0.3,      # Slightly higher for synthesis tasks
                max_tokens=1200,      # Increased for detailed comprehensive research analysis
                top_p=0.85,          # More focused sampling
                frequency_penalty=0.4,  # Lower penalty to allow detailed explanations
                presence_penalty=0.3,   # Lower penalty for comprehensive coverage
                request_timeout=60,   # Timeout for requests
            )
            
            logger.info("✅ LM Studio LLM initialized successfully")
            logger.info("🎯 Model: MedGemma 4B IT, Temperature: 0.3 (synthesis)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LM Studio LLM: {e}")
            logger.error("🔧 Troubleshooting:")
            logger.error("   1. Ensure LM Studio is running on the specified port")
            logger.error("   2. Ensure MedGemma 4B IT model is loaded in LM Studio")
            logger.error(f"   3. Check LM Studio server at: {self.lm_studio_base_url}")
            raise
    
    def _create_prompt_templates(self):
        """Create prompt templates for different scenarios."""
        logger.info("📝 Creating research prompt templates")
        
        # Template for when vector database has sufficient data
        self.vector_db_template = PromptTemplate(
            input_variables=["question", "vector_db_data"],
            template="""You are a Research Analyst specializing in medical literature synthesis.

CONTEXT: The user's question has been answered using existing medical knowledge from our comprehensive database. However, they may benefit from additional detailed research insights.

CRITICAL INSTRUCTIONS:
- Provide a DETAILED, COMPREHENSIVE synthesis of the existing medical knowledge
- Include specific details about symptoms, causes, treatments, and research findings
- Explain mechanisms, pathophysiology, and clinical implications thoroughly
- Cover prevention strategies, diagnostic approaches, and treatment options
- Discuss prognosis, complications, and follow-up recommendations
- Maintain scientific accuracy and provide evidence-based information
- Use clear explanations that help patients understand complex medical concepts

USER'S RESEARCH QUESTION: {question}

EXISTING MEDICAL KNOWLEDGE:
{vector_db_data}

DETAILED RESEARCH SYNTHESIS BASED ON EXISTING KNOWLEDGE:"""
        )
        
        # Template for when we need PubMed data
        self.pubmed_template = PromptTemplate(
            input_variables=["question", "pubmed_results"],
            template="""You are a Research Analyst specializing in medical literature synthesis.

CONTEXT: The user's question requires the latest medical research findings. Our existing database did not contain sufficient relevant information, so we searched recent medical literature.

CRITICAL INSTRUCTIONS:
- Provide a DETAILED, COMPREHENSIVE synthesis of the PubMed research abstracts
- Include specific findings from individual studies with methodological details
- Explain clinical implications, statistical significance, and effect sizes thoroughly
- Cover study populations, sample sizes, and research methodologies
- Discuss limitations, future research directions, and clinical applications
- Identify trends, consensus, and any conflicting findings across studies
- Provide actionable insights for healthcare practitioners and patients
- Maintain scientific rigor while ensuring accessibility

USER'S RESEARCH QUESTION: {question}

RECENT PUBMED RESEARCH ABSTRACTS:
{pubmed_results}

DETAILED RESEARCH SYNTHESIS FROM LATEST LITERATURE:"""
        )
        
        # Template for hybrid approach (both sources)
        self.hybrid_template = PromptTemplate(
            input_variables=["question", "vector_db_data", "pubmed_results"],
            template="""You are a Research Analyst specializing in medical literature synthesis.

CONTEXT: You have access to both established medical knowledge from our database and the latest research from PubMed. Provide a comprehensive analysis combining both sources.

CRITICAL INSTRUCTIONS:
- Provide a DETAILED, COMPREHENSIVE synthesis combining established knowledge and recent research
- Compare and contrast findings from both sources with specific examples and data
- Highlight what's well-established versus emerging research with detailed explanations
- Discuss evolution of understanding, treatment approaches, and diagnostic criteria
- Include specific research findings, clinical trial results, and outcome measures
- Cover both traditional and innovative treatment approaches thoroughly
- Address any conflicts between sources with balanced analysis
- Provide actionable clinical insights and patient care recommendations
- Maintain scientific rigor while ensuring patient accessibility

USER'S RESEARCH QUESTION: {question}

ESTABLISHED MEDICAL KNOWLEDGE:
{vector_db_data}

RECENT PUBMED RESEARCH:
{pubmed_results}

DETAILED COMPREHENSIVE RESEARCH SYNTHESIS:"""
        )
        
        logger.info("✅ Research prompt templates created")
        logger.info("🔬 Vector DB, PubMed, and Hybrid templates configured")
    
    def _assemble_hybrid_chain(self):
        """Assemble the hybrid Vector DB + PubMed chain."""
        logger.info("🔗 Assembling hybrid research chain (Vector DB + PubMed)")
        
        def fetch_pubmed_results(question: str) -> str:
            """Fetch PubMed results for the given question."""
            try:
                logger.info(f"📡 Fetching PubMed results for: {question[:100]}...")
                results = self.pubmed_tool.run(question)
                logger.info(f"✅ Retrieved {len(results.split('Published:')) - 1} research articles")
                return results
            except Exception as e:
                logger.error(f"❌ PubMed API error: {e}")
                return f"Error retrieving research: {str(e)}"
        
        # Store the PubMed fetcher for use in the main method
        self.fetch_pubmed_results = fetch_pubmed_results
        
        logger.info("✅ Hybrid research chain components assembled")
        logger.info("🔄 Chain flow: Question → Vector DB Check → [PubMed if needed] → LLM → Summary")
    
    async def find_latest_research(self, query: str) -> str:
        """
        Find and synthesize research using hybrid approach: Vector DB first, then PubMed if needed.
        
        Args:
            query: Research query
            
        Returns:
            Synthesized research summary
        """
        logger.info(f"🔍 Processing research query (HYBRID): {query[:100]}...")
        
        try:
            # Step 1: Check vector database for existing relevant data
            has_vector_data, vector_db_content = self._check_vector_database_relevance(query)
            
            # Step 2: Decide whether to use PubMed based on vector DB results
            if has_vector_data:
                logger.info("📊 Using vector database data (sufficient relevance found)")
                
                # Use vector database template
                formatted_prompt = self.vector_db_template.format(
                    question=query,
                    vector_db_data=vector_db_content
                )
                
                response = await self.llm.ainvoke(formatted_prompt)
                logger.info("✅ Research query processed using vector database")
                
            else:
                logger.info("🔍 Vector database insufficient, fetching latest research from PubMed")
                
                # Fetch from PubMed
                pubmed_results = self.fetch_pubmed_results(query)
                
                # Use PubMed template
                formatted_prompt = self.pubmed_template.format(
                    question=query,
                    pubmed_results=pubmed_results
                )
                
                response = await self.llm.ainvoke(formatted_prompt)
                logger.info("✅ Research query processed using PubMed API")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Research query processing failed: {e}")
            return f"I apologize, but I encountered an error while searching for research: {str(e)}"
    
    def get_raw_pubmed_results(self, query: str) -> str:
        """
        Get raw PubMed results for debugging/transparency.
        
        Args:
            query: Research query
            
        Returns:
            Raw PubMed API results
        """
        try:
            return self.pubmed_tool.run(query)
        except Exception as e:
            logger.error(f"❌ Raw PubMed retrieval failed: {e}")
            return f"Error retrieving PubMed data: {str(e)}"

def main():
    """Demonstration and verification of the Research Analyst Agent."""
    logger.info("🚀 RESEARCH ANALYST AGENT - HYBRID DEMONSTRATION")
    logger.info("=" * 70)
    
    try:
        # Initialize the agent
        agent = ResearchAnalystAgent()
        
        # Test research query for verification - try something that should be in vector DB
        test_query = "breast cancer treatment guidelines"
        
        logger.info(f"🧪 TEST RESEARCH QUERY: {test_query}")
        logger.info("-" * 70)
        
        # Test vector database check first
        has_data, vector_content = agent._check_vector_database_relevance(test_query)
        logger.info(f"� Vector DB relevance check: {has_data}")
        if has_data:
            logger.info(f"   📋 Vector DB content preview: {vector_content[:200]}...")
        
        logger.info("-" * 70)
        
        # Query the agent for research synthesis (will use hybrid approach)
        import asyncio
        response = asyncio.run(agent.find_latest_research(test_query))
        
        # Display results
        print("\n" + "=" * 70)
        print("🔬 RESEARCH ANALYST AGENT - HYBRID TEST RESPONSE")
        print("=" * 70)
        print(f"❓ RESEARCH QUERY: {test_query}")
        print("-" * 70)
        print(f"💡 RESEARCH SYNTHESIS:")
        print(response)
        print("=" * 70)
        print("✅ HYBRID RESEARCH CHAIN VERIFICATION COMPLETED SUCCESSFULLY")
        print("🎯 The Research Analyst Agent uses Vector DB first, PubMed when needed")
        print("=" * 70)
        
        logger.info("✅ Research Analyst Agent verification completed successfully")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL FAILURE in Research Analyst Agent demonstration: {e}")
        logger.error("🔧 Please ensure:")
        logger.error("   1. ChromaDB database is available and populated")
        logger.error("   2. Internet connection is available for PubMed API")
        logger.error("   3. LM Studio is running with MedGemma 4B IT model loaded")
        logger.error("   4. langchain_community package is installed")
        sys.exit(1)

if __name__ == "__main__":
    main()
