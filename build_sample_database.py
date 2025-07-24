"""
Sample Database Builder for MedIntellect
=======================================

This script creates a small sample database for testing the MedIntellect system
without requiring the full medical datasets. Use this for development and demonstration.

⚠️ IMPORTANT: This is for testing only. For production use, build a proper
medical knowledge database using real medical literature.
"""

import chromadb
from chromadb.config import Settings
import os
import sys
from typing import List, Dict

def create_sample_medical_data() -> Dict[str, List[Dict]]:
    """Create sample medical data for testing."""
    
    sample_guidelines = [
        {
            "id": "guideline_1",
            "text": "Diabetes Management Guidelines: Type 2 diabetes should be managed through lifestyle modifications including diet and exercise. Initial medication typically includes metformin unless contraindicated. Blood glucose targets should be individualized but generally HbA1c <7% for most adults.",
            "metadata": {"source": "sample_diabetes_guidelines", "type": "guideline"}
        },
        {
            "id": "guideline_2", 
            "text": "Hypertension Treatment Protocol: Blood pressure targets are <130/80 mmHg for most adults. First-line treatments include ACE inhibitors, ARBs, thiazide diuretics, or calcium channel blockers. Lifestyle modifications should always be recommended.",
            "metadata": {"source": "sample_hypertension_guidelines", "type": "guideline"}
        },
        {
            "id": "guideline_3",
            "text": "Breast Cancer Screening Guidelines: Women aged 50-74 should undergo mammography screening every 2 years. Women with family history or genetic predisposition may need earlier or more frequent screening. Clinical breast examination should be performed annually.",
            "metadata": {"source": "sample_cancer_guidelines", "type": "guideline"}
        },
        {
            "id": "guideline_4",
            "text": "COVID-19 Management Protocol: Mild cases can be managed at home with supportive care. Severe cases require hospitalization. Antiviral treatments may be considered for high-risk patients. Vaccination remains the primary prevention strategy.",
            "metadata": {"source": "sample_covid_guidelines", "type": "guideline"}
        },
        {
            "id": "guideline_5",
            "text": "Heart Disease Prevention Guidelines: Primary prevention includes lifestyle modifications: smoking cessation, regular exercise, healthy diet, and weight management. Statin therapy may be indicated for patients with elevated cardiovascular risk.",
            "metadata": {"source": "sample_cardiology_guidelines", "type": "guideline"}
        }
    ]
    
    sample_qna = [
        {
            "id": "qna_1",
            "text": "Question: What are the common symptoms of diabetes?\nAnswer: Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Type 1 diabetes symptoms often develop quickly, while Type 2 diabetes symptoms may develop gradually.",
            "metadata": {"source": "sample_diabetes_qna", "type": "qna"}
        },
        {
            "id": "qna_2",
            "text": "Question: How is high blood pressure diagnosed?\nAnswer: High blood pressure is diagnosed through multiple blood pressure readings taken on separate occasions. A diagnosis of hypertension is made when systolic pressure is consistently ≥130 mmHg or diastolic pressure is ≥80 mmHg. Ambulatory blood pressure monitoring may be used for confirmation.",
            "metadata": {"source": "sample_hypertension_qna", "type": "qna"}
        },
        {
            "id": "qna_3",
            "text": "Question: What are the risk factors for breast cancer?\nAnswer: Risk factors for breast cancer include age (most common after 50), family history, genetic mutations (BRCA1/BRCA2), personal history of breast disease, radiation exposure, hormone replacement therapy, alcohol consumption, obesity, and reproductive factors like early menstruation or late menopause.",
            "metadata": {"source": "sample_cancer_qna", "type": "qna"}
        },
        {
            "id": "qna_4",
            "text": "Question: What are the side effects of chemotherapy?\nAnswer: Common chemotherapy side effects include nausea and vomiting, hair loss, fatigue, increased infection risk due to low white blood cell count, easy bruising or bleeding, mouth sores, diarrhea or constipation, appetite changes, and neuropathy. Side effects vary by drug type and individual patient factors.",
            "metadata": {"source": "sample_treatment_qna", "type": "qna"}
        },
        {
            "id": "qna_5",
            "text": "Question: How can heart disease be prevented?\nAnswer: Heart disease prevention includes maintaining a healthy diet low in saturated fats and sodium, regular physical activity, maintaining a healthy weight, not smoking, limiting alcohol consumption, managing stress, controlling blood pressure and cholesterol levels, and managing diabetes if present.",
            "metadata": {"source": "sample_prevention_qna", "type": "qna"}
        },
        {
            "id": "qna_6",
            "text": "Question: What vaccines are recommended for adults?\nAnswer: Adult vaccines typically include annual influenza vaccine, Tdap booster every 10 years, pneumococcal vaccine for adults 65+, shingles vaccine for adults 50+, HPV vaccine for adults up to age 26, and COVID-19 vaccines as recommended. Additional vaccines may be needed based on travel, occupation, or medical conditions.",
            "metadata": {"source": "sample_vaccination_qna", "type": "qna"}
        }
    ]
    
    return {
        "guidelines": sample_guidelines,
        "qna": sample_qna
    }

def build_sample_database():
    """Build a sample ChromaDB database for testing."""
    
    print("🏥 Building Sample MedIntellect Database")
    print("=" * 50)
    
    # Create database directory
    db_path = "./chroma_db_sample"
    os.makedirs(db_path, exist_ok=True)
    
    try:
        # Initialize ChromaDB
        print("📀 Initializing ChromaDB...")
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get sample data
        print("📝 Creating sample medical data...")
        sample_data = create_sample_medical_data()
        
        # Create guidelines collection
        print("🏥 Creating medical_guidelines collection...")
        guidelines_collection = client.get_or_create_collection(
            name="medical_guidelines",
            metadata={"description": "Sample medical guidelines and protocols"}
        )
        
        # Add guidelines data
        guidelines_texts = [item["text"] for item in sample_data["guidelines"]]
        guidelines_ids = [item["id"] for item in sample_data["guidelines"]]
        guidelines_metadata = [item["metadata"] for item in sample_data["guidelines"]]
        
        guidelines_collection.add(
            documents=guidelines_texts,
            ids=guidelines_ids,
            metadatas=guidelines_metadata
        )
        
        print(f"   ✅ Added {len(guidelines_texts)} guideline documents")
        
        # Create Q&A collection
        print("🩺 Creating medrag_qna collection...")
        qna_collection = client.get_or_create_collection(
            name="medrag_qna",
            metadata={"description": "Sample medical Q&A pairs"}
        )
        
        # Add Q&A data
        qna_texts = [item["text"] for item in sample_data["qna"]]
        qna_ids = [item["id"] for item in sample_data["qna"]]
        qna_metadata = [item["metadata"] for item in sample_data["qna"]]
        
        qna_collection.add(
            documents=qna_texts,
            ids=qna_ids,
            metadatas=qna_metadata
        )
        
        print(f"   ✅ Added {len(qna_texts)} Q&A documents")
        
        # Verify collections
        print("\n📊 Database Summary:")
        print(f"   📁 Database path: {db_path}")
        print(f"   🏥 Guidelines collection: {guidelines_collection.count()} documents")
        print(f"   🩺 Q&A collection: {qna_collection.count()} documents")
        print(f"   📈 Total documents: {guidelines_collection.count() + qna_collection.count()}")
        
        print("\n✅ Sample database created successfully!")
        print("\n🚀 Next steps:")
        print("   1. Update your medical_system.py to use ./chroma_db_sample")
        print("   2. Run: streamlit run streamlit_app.py")
        print("   3. Try sample questions like 'What are the symptoms of diabetes?'")
        
        print("\n⚠️  REMINDER: This is sample data for testing only.")
        print("   For production use, build a proper medical knowledge database.")
        
    except Exception as e:
        print(f"❌ Error building sample database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_sample_database()
