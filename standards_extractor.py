import os
import json
import PyPDF2
from transformers import pipeline

class StandardsExtractor:
    def __init__(self, hf_api_token=None):
        """Initialize the standards extractor with an optional HF API token"""
        self.hf_api_token = hf_api_token
        # Initialize the extractive QA pipeline - will use API if token provided
        if hf_api_token:
            self.qa_pipeline = None  # Will use API calls
        else:
            # Use smaller model that can run on CPU
            self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    
    def extract_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None

    def extract_standard_elements(self, text, standard_type="honey"):
        """Extract standardized elements based on the standard type"""
        if standard_type == "honey":
            # Define extraction questions for honey standards
            questions = {
                "moisture_content": "What is the maximum allowed moisture content for honey?",
                "hMF_content": "What is the maximum allowed hydroxymethylfurfural (HMF) content?",
                "diastase_activity": "What is the minimum diastase activity required?",
                "sucrose_content": "What is the maximum allowed sucrose content?",
                "free_acidity": "What is the maximum allowed free acidity?",
                "electrical_conductivity": "What are the requirements for electrical conductivity?",
                "insoluble_solids": "What are the limits for water insoluble solids?",
                "glucose_fructose": "What are the requirements for total glucose and fructose content?"
            }
            
            # Extract answers using QA pipeline
            standard_elements = {}
            for key, question in questions.items():
                answer = self._get_answer(text, question)
                standard_elements[key] = answer
                
            return standard_elements
        else:
            # Add support for other standard types as needed
            return {}
    
    def _get_answer(self, context, question):
        """Get answer using QA pipeline or API"""
        if self.qa_pipeline:
            # Use local pipeline
            result = self.qa_pipeline(question=question, context=context)
            return result['answer']
        else:
            # Use Hugging Face Inference API
            import requests
            API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
            headers = {"Authorization": f"Bearer {self.hf_api_token}"}
            payload = {
                "inputs": {
                    "question": question,
                    "context": context
                }
            }
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json().get('answer', '')
            else:
                print(f"API Error: {response.status_code}")
                return ""
    
    def save_to_json(self, standard_elements, output_path):
        """Save extracted standard elements to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(standard_elements, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            return False
    
    def process_standard_document(self, pdf_path, output_json_path, standard_type="honey"):
        """Process a standard document and save extracted elements to JSON"""
        text = self.extract_from_pdf(pdf_path)
        if text:
            elements = self.extract_standard_elements(text, standard_type)
            return self.save_to_json(elements, output_json_path)
        return False
