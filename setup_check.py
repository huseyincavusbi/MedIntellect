#!/usr/bin/env python3
"""
MedIntellect Setup Verification Script
====================================

This script verifies that your MedIntellect system is properly configured
and ready to run. Run this after installation to ensure everything works.

Usage: python setup_check.py
"""

import sys
import os
import requests
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🔍 {text}")
    print(f"{'='*60}")

def print_check(description, status, details=""):
    """Print a formatted check result."""
    icon = "✅" if status else "❌"
    print(f"{icon} {description}")
    if details:
        print(f"   {details}")

def check_python_version():
    """Check Python version compatibility."""
    print_header("Python Environment Check")
    
    version = sys.version_info
    is_compatible = version.major == 3 and version.minor >= 8
    
    print_check(
        f"Python Version: {version.major}.{version.minor}.{version.micro}",
        is_compatible,
        "✅ Compatible" if is_compatible else "❌ Requires Python 3.8+"
    )
    
    return is_compatible

def check_required_packages():
    """Check if required packages are installed."""
    print_header("Package Dependencies Check")
    
    required_packages = [
        "streamlit",
        "langchain",
        "chromadb",
        "sentence_transformers",
        "numpy",
        "pandas"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print_check(f"{package}", True, "Installed")
        except ImportError:
            print_check(f"{package}", False, "Missing - run: pip install -r requirements.txt")
            all_installed = False
    
    return all_installed

def check_lm_studio():
    """Check if LM Studio is running and accessible."""
    print_header("LM Studio Connection Check")
    
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print_check("LM Studio Connection", True, f"Connected on localhost:1234")
            
            if models.get('data'):
                model_names = [model.get('id', 'Unknown') for model in models['data']]
                print_check("Loaded Models", True, f"Found: {', '.join(model_names)}")
                
                # Check for MedGemma specifically
                has_medgemma = any('medgemma' in model.lower() for model in model_names)
                print_check("MedGemma Model", has_medgemma, 
                          "MedGemma found" if has_medgemma else "Load MedGemma 4B IT MLX in LM Studio")
                
                return True
            else:
                print_check("Loaded Models", False, "No models loaded in LM Studio")
                return False
                
        else:
            print_check("LM Studio Connection", False, f"HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print_check("LM Studio Connection", False, "Not running - start LM Studio server")
        return False

def check_database():
    """Check if ChromaDB database exists."""
    print_header("Database Check")
    
    db_paths = [
        "./chroma_db_test/chroma_db",
        "./chroma_db",
        "./chroma_db_sample"
    ]
    
    found_db = False
    for db_path in db_paths:
        if os.path.exists(db_path):
            print_check(f"Database: {db_path}", True, "Found")
            found_db = True
            break
    
    if not found_db:
        print_check("Database", False, "Run: python build_sample_database.py")
    
    return found_db

def check_core_files():
    """Check if core system files exist."""
    print_header("Core Files Check")
    
    core_files = [
        "streamlit_app_fixed.py",
        "medical_system.py",
        "generalist_doctor_agent.py",
        "guideline_specialist_agent.py",
        "research_analyst_agent.py"
    ]
    
    all_present = True
    for file in core_files:
        exists = os.path.exists(file)
        print_check(file, exists)
        if not exists:
            all_present = False
    
    return all_present

def run_quick_test():
    """Run a quick test of core functionality."""
    print_header("Quick Functionality Test")
    
    try:
        # Test embedding model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedding = model.encode("test medical query")
        print_check("Embedding Generation", True, f"Vector dim: {len(embedding)}")
        
        # Test ChromaDB import
        import chromadb
        print_check("ChromaDB Import", True, "Ready")
        
        # Test LangChain import
        from langchain.schema import Document
        print_check("LangChain Import", True, "Ready")
        
        return True
        
    except Exception as e:
        print_check("Functionality Test", False, f"Error: {str(e)}")
        return False

def main():
    """Run all verification checks."""
    print("🏥 MedIntellect Setup Verification")
    print("=" * 60)
    print("This script checks if your system is ready to run MedIntellect.")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("LM Studio", check_lm_studio),
        ("Database", check_database),
        ("Core Files", check_core_files),
        ("Quick Test", run_quick_test)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        if check_func():
            passed += 1
    
    # Final summary
    print_header("Setup Verification Summary")
    
    if passed == total:
        print("🎉 All checks passed! Your MedIntellect system is ready to run.")
        print("\n📋 Next Steps:")
        print("   1. Start LM Studio with MedGemma 4B IT MLX")
        print("   2. Run: streamlit run streamlit_app_fixed.py")
        print("   3. Open: http://localhost:8501")
        
    else:
        print(f"⚠️  {passed}/{total} checks passed. Please fix the issues above.")
        print("\n🔧 Common Solutions:")
        print("   • Install packages: pip install -r requirements.txt")
        print("   • Start LM Studio and load MedGemma model")
        print("   • Create database: python build_sample_database.py")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
