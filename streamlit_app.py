"""
MedIntellect - Streamlit Web Interface
==========================================
"""

import sys
import streamlit as st
import asyncio
import time
import os
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from medical_system import AsyncMedicalSystem, MedicalSystem

# Page configuration
st.set_page_config(
    page_title="🏥 MedIntellect",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better dark mode support
st.markdown("""
<style>
/* Medical consultation result styling - WHITE TEXT */
.medical-result {
    background: rgba(34, 197, 94, 0.1) !important;
    padding: 1.5rem !important;
    border-radius: 0.75rem !important;
    border: 2px solid rgba(34, 197, 94, 0.3) !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    margin: 1rem 0 !important;
}

/* WHITE TEXT for all medical results */
.medical-result p {
    margin: 0 !important;
    line-height: 1.6 !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    color: #ffffff !important;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5) !important;
}

/* Dark mode specific styling */
@media (prefers-color-scheme: dark) {
    .medical-result {
        background: rgba(34, 197, 94, 0.15) !important;
        border-color: rgba(34, 197, 94, 0.4) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
}

/* Streamlit dark theme detection */
.stApp[data-theme="dark"] .medical-result {
    background: rgba(34, 197, 94, 0.15) !important;
    border-color: rgba(34, 197, 94, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'medical_system' not in st.session_state:
    st.session_state.medical_system = None
if 'consultation_history' not in st.session_state:
    st.session_state.consultation_history = []
if 'follow_up_mode' not in st.session_state:
    st.session_state.follow_up_mode = False
if 'current_consultation' not in st.session_state:
    st.session_state.current_consultation = None

def initialize_sync_system():
    """Initialize the medical system synchronously."""
    try:
        system = MedicalSystem()
        return system
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return None

def initialize_async_system():
    """Initialize the medical system asynchronously."""
    try:
        # Simple async handling for Streamlit
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def create_system():
            return AsyncMedicalSystem()
        
        system = loop.run_until_complete(create_system())
        loop.close()
        return system
    except Exception as e:
        st.error(f"Failed to initialize async: {e}")
        return None

# Main header
st.title("🏥 MedIntellect - Medical AI Assistant")

# Sidebar
with st.sidebar:
    st.header("🔧 System Configuration")
    
    # System initialization
    if not st.session_state.system_initialized:
        st.info("⚡ Choose initialization method:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Async", use_container_width=True):
                with st.spinner("Initializing..."):
                    result = initialize_async_system()
                    if result:
                        st.session_state.medical_system = result
                        st.session_state.system_initialized = True
                        st.success("✅ Async system ready!")
                        st.rerun()
                    else:
                        st.error("❌ Async failed")
        
        with col2:
            if st.button("🔧 Sync", use_container_width=True):
                with st.spinner("Initializing..."):
                    result = initialize_sync_system()
                    if result:
                        st.session_state.medical_system = result
                        st.session_state.system_initialized = True
                        st.success("✅ Sync system ready!")
                        st.rerun()
                    else:
                        st.error("❌ Sync failed")
    else:
        st.success("✅ Medical System Ready")
        
        if st.button("🔄 Reset System"):
            st.session_state.system_initialized = False
            st.session_state.medical_system = None
            st.session_state.consultation_history = []
            st.session_state.follow_up_mode = False
            st.session_state.current_consultation = None
            st.rerun()
    
    st.divider()
    
    # Show consultation history count
    if st.session_state.consultation_history:
        st.write(f"📊 Consultations: {len(st.session_state.consultation_history)}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.consultation_history = []
            st.session_state.follow_up_mode = False
            st.session_state.current_consultation = None
            st.rerun()

# Main content
if not st.session_state.system_initialized:
    st.info("👈 Please initialize the medical system using the sidebar.")
    
    # System info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### ⚡ Parallel Processing
        Multiple specialist agents work simultaneously
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Evidence-Based
        All responses backed by medical literature
        """)
    
    with col3:
        st.markdown("""
        ### 🔒 Anti-Hallucination
        Strict controls prevent false information
        """)

else:
    # Consultation interface
    st.header("💬 Medical Consultation")
    
    # Follow-up mode indicator
    if st.session_state.follow_up_mode and st.session_state.current_consultation:
        st.info(f"🔄 **Follow-up Mode** - Previous: {st.session_state.current_consultation['question'][:50]}...")
        if st.button("🆕 Start New Consultation"):
            st.session_state.follow_up_mode = False
            st.session_state.current_consultation = None
            st.rerun()
    
    # Question input
    question = st.text_area(
        "Enter your medical question:",
        placeholder="What are the symptoms and risk factors of breast cancer?",
        height=100,
        key="question_input"
    )
    
    # Submit button
    if st.button("🔍 Get Medical Consultation", type="primary", disabled=not question.strip()):
        if question.strip():
            with st.spinner("🔄 Consulting medical specialists..."):
                start_time = time.time()
                
                try:
                    # Prepare question
                    consultation_question = question.strip()
                    
                    # Auto-detect new questions to exit follow-up mode
                    if st.session_state.follow_up_mode:
                        new_indicators = ['what are the symptoms', 'what is', 'how is', 'tell me about', 'explain', 'describe']
                        if any(indicator in consultation_question.lower() for indicator in new_indicators):
                            st.session_state.follow_up_mode = False
                            st.session_state.current_consultation = None
                            st.info("🆕 Detected new question - starting fresh consultation")
                    
                    # Add context for follow-ups
                    if st.session_state.follow_up_mode and st.session_state.current_consultation:
                        prev_q = st.session_state.current_consultation['question']
                        prev_a = st.session_state.current_consultation['result'].get('final_answer', '')
                        
                        consultation_question = f"""Previous consultation:
Question: {prev_q}
Answer: {prev_a[:300]}...

Follow-up question: {question.strip()}

Please answer the follow-up question considering the previous context."""
                    
                    # Show debug info
                    with st.expander("🐛 Debug Info", expanded=False):
                        st.write(f"**Follow-up mode:** {st.session_state.follow_up_mode}")
                        st.write(f"**Question being sent:**")
                        st.code(consultation_question)
                    
                    # Execute consultation
                    if hasattr(st.session_state.medical_system.consult, '__call__'):
                        # Check if it's async
                        if asyncio.iscoroutinefunction(st.session_state.medical_system.consult):
                            # Async system
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(st.session_state.medical_system.consult(consultation_question))
                            loop.close()
                        else:
                            # Sync system
                            result = st.session_state.medical_system.consult(consultation_question)
                    else:
                        st.error("❌ Invalid medical system")
                        result = None
                    
                    end_time = time.time()
                    consultation_time = end_time - start_time
                    
                    if result:
                        # Store in history
                        consultation_record = {
                            'question': question.strip(),
                            'original_question': consultation_question,
                            'result': result,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'consultation_time': consultation_time,
                            'is_follow_up': st.session_state.follow_up_mode
                        }
                        st.session_state.consultation_history.append(consultation_record)
                        st.session_state.current_consultation = consultation_record
                        
                        # Display results
                        st.success(f"✅ Consultation completed in {consultation_time:.2f} seconds")
                        
                        # Show final answer
                        st.subheader("💡 Medical Consultation Result")
                        final_answer = result.get('final_answer', 'No response generated')
                        
                        # Check for vaccine response issue
                        if "vaccine" in final_answer.lower() and "vaccine" not in question.lower():
                            st.error("⚠️ WARNING: Detected potential irrelevant vaccine response!")
                            st.info("💡 Try clicking 'Start New Consultation' and ask your question again")
                        
                        st.markdown(f"""
                        <div class="medical-result">
                            <p>{final_answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Enable follow-up mode
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 Ask Follow-up Question"):
                                st.session_state.follow_up_mode = True
                                st.rerun()
                        with col2:
                            if st.button("🆕 New Consultation"):
                                st.session_state.follow_up_mode = False
                                st.session_state.current_consultation = None
                                st.rerun()
                    
                    else:
                        st.error("❌ Consultation failed - no result returned")
                
                except Exception as e:
                    st.error(f"❌ Consultation error: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    # Recent consultations
    if st.session_state.consultation_history:
        st.divider()
        st.subheader("📋 Recent Consultations")
        
        for i, record in enumerate(reversed(st.session_state.consultation_history[-3:])):
            follow_up_indicator = "🔄 " if record.get('is_follow_up', False) else "💬 "
            
            with st.expander(f"{follow_up_indicator}{record['timestamp']} - {record['question'][:50]}..."):
                st.write(f"**Question:** {record['question']}")
                st.write(f"**Answer:** {record['result'].get('final_answer', 'No answer')[:200]}...")
                
                if st.button(f"🔄 Ask follow-up", key=f"followup_{i}"):
                    st.session_state.current_consultation = record
                    st.session_state.follow_up_mode = True
                    st.rerun()
