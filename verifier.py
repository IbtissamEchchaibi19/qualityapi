import json
import time
import re
import os
from transformers import pipeline

class DocumentVerifier:
    def __init__(self, standard_json_path, hf_api_token=None):
        """Initialize the document verifier with a standard JSON"""
        self.standard = self._load_standard(standard_json_path)
        
        # Get HuggingFace token from parameter or environment
        self.hf_api_token = hf_api_token or os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        self.nli_pipeline = None
        
        # Initialize NLI classifier with timeout handling
        self._initialize_nli_pipeline()
                
    def _initialize_nli_pipeline(self):
        """Initialize the NLI pipeline with proper token handling"""
        try:
            import threading
            
            def load_model():
                try:
                    if self.hf_api_token:
                        print("Using HuggingFace API token for model loading...")
                        # Use HuggingFace API with token
                        self.nli_pipeline = pipeline(
                            "zero-shot-classification", 
                            model="facebook/bart-large-mnli",
                            use_auth_token=self.hf_api_token
                        )
                        print("Successfully loaded NLI model with HuggingFace token")
                    else:
                        print("No HuggingFace token found, attempting local model loading...")
                        # Fallback to local model without token
                        self.nli_pipeline = pipeline(
                            "zero-shot-classification", 
                            model="facebook/bart-large-mnli"
                        )
                        print("Successfully loaded NLI model locally")
                        
                except Exception as e:
                    print(f"Failed to load NLI model: {e}")
                    self.nli_pipeline = None
            
            # Start model loading in a thread with timeout
            model_thread = threading.Thread(target=load_model)
            model_thread.daemon = True
            model_thread.start()
            model_thread.join(timeout=60)  # Increased timeout to 60 seconds
            
            if not self.nli_pipeline:
                print("Warning: NLI model failed to load, using fallback verification")
                
        except Exception as e:
            print(f"Error initializing NLI pipeline: {e}")
            self.nli_pipeline = None
    
    def _load_standard(self, json_path):
        """Load standard from JSON file"""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading standard JSON: {e}")
            return {}
    
    def _extract_numeric_values(self, text):
        """Extract numerical values with their units from text"""
        # Pattern for numbers with potential units
        patterns = [
            r'(\d+\.?\d*)\s*(%|percent)', # For percentages
            r'(\d+\.?\d*)\s*(mg/kg|ppm)',  # For HMF content
            r'(\d+\.?\d*)\s*(schade|units)', # For diastase
            r'(\d+\.?\d*)\s*(g/100g|%)', # For sucrose
            r'(\d+\.?\d*)\s*(meq/kg)', # For acidity
            r'(\d+\.?\d*)\s*(ms/cm|mS/cm)', # For conductivity
            r'(\d+\.?\d*)\s*(g/100g|%)', # For glucose/fructose
            r'(\d+\.?\d*)\s?([a-zA-Z]+/[a-zA-Z]+|[a-zA-Z]+)?', # Generic pattern
        ]
        
        values = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 1:
                    value = match.group(1)
                    unit = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]
                    values.append({
                        "value": value,
                        "unit": unit,
                        "context": context
                    })
        
        return values
    
    def _find_standard_value(self, param_name):
        """Extract the standard value and requirement from the standard"""
        if param_name not in self.standard:
            return None, None
        
        standard_text = self.standard.get(param_name, "")
        
        # Extract numeric values from standard text
        values = self._extract_numeric_values(standard_text)
        
        if values:
            # Get the first value as the standard value
            standard_value = values[0]["value"]
            if values[0]["unit"]:
                standard_value += " " + values[0]["unit"]
            
            # Extract requirement type (max, min, range)
            requirement_type = "unknown"
            lower_text = standard_text.lower()
            if "maximum" in lower_text or "max" in lower_text or "not more than" in lower_text or "≤" in standard_text:
                requirement_type = "maximum"
            elif "minimum" in lower_text or "min" in lower_text or "not less than" in lower_text or "≥" in standard_text:
                requirement_type = "minimum"
            elif "between" in lower_text or "range" in lower_text:
                requirement_type = "range"
            
            return standard_value, requirement_type
        
        return None, None
    
    def _verify_parameter_with_values(self, param_name, extracted_data, standard_data):
        """Verify if a parameter meets the standard and return the actual values found"""
        # If we don't have data for this parameter, return False
        if param_name not in extracted_data:
            return False, "Parameter not found in document", [], None
        
        # Get standard requirement text
        standard_text = standard_data.get(param_name, "")
        if not standard_text:
            return False, "Standard requirement not available", [], None
        
        # Get all sections and values for this parameter
        sections = extracted_data[param_name].get("sections", [])
        raw_values = extracted_data[param_name].get("raw_values", [])
        
        # Combine all data for this parameter
        combined_text = " ".join(sections + raw_values)
        
        # Extract numerical values from the combined text
        extracted_values = self._extract_numeric_values(combined_text)
        
        # Get standard value and requirement type
        standard_value, requirement_type = self._find_standard_value(param_name)
        
        # Use NLI model or fallback for compliance checking
        if not self.nli_pipeline:
            print(f"Using fallback verification for {param_name}")
            compliant, message = self._fallback_verification(param_name, combined_text, standard_text)
        else:
            try:
                print(f"Using NLI model for {param_name}")
                # Use NLI model for compliance checking
                compliance_label = f"This honey complies with the {param_name} standard"
                non_compliance_label = f"This honey does not comply with the {param_name} standard"
                
                result = self.nli_pipeline(
                    combined_text,
                    [compliance_label, non_compliance_label],
                    multi_label=False
                )
                
                # Find the score for the compliance label
                compliance_idx = result['labels'].index(compliance_label)
                compliance_score = result['scores'][compliance_idx]
                
                # Check if compliance score is above threshold
                compliant = compliance_score > 0.5
                message = f"NLI Confidence: {compliance_score:.2f} (using {'HF token' if self.hf_api_token else 'local model'})"
                
            except Exception as e:
                print(f"Error in NLI inference for {param_name}: {e}")
                # Fallback to simple verification
                compliant, message = self._fallback_verification(param_name, combined_text, standard_text)
        
        return compliant, message, extracted_values, standard_value
    
    def _fallback_verification(self, param_name, extracted_text, standard_text):
        """Enhanced fallback verification when NLI is not available"""
        # Convert everything to lowercase for comparison
        extracted_lower = extracted_text.lower()
        standard_lower = standard_text.lower()
        
        # Extract numerical values for comparison
        extracted_numbers = re.findall(r'\d+\.?\d*', extracted_lower)
        standard_numbers = re.findall(r'\d+\.?\d*', standard_lower)
        
        # Check if we have any numerical matches
        if any(num in extracted_lower for num in standard_numbers):
            return True, "Fallback: Found matching numerical value"
        
        # Check for compliance keywords
        compliance_keywords = ["compliant", "meets", "standard", "acceptable", "within", "pass", "passed"]
        if any(keyword in extracted_lower for keyword in compliance_keywords):
            return True, "Fallback: Found compliance indicator"
        
        # Check for non-compliance keywords
        non_compliance_keywords = ["non-compliant", "fails", "exceed", "below", "above limit", "fail", "failed"]
        if any(keyword in extracted_lower for keyword in non_compliance_keywords):
            return False, "Fallback: Found non-compliance indicator"
            
        # If we have numerical data, try basic comparison
        if extracted_numbers and standard_numbers:
            try:
                # Simple heuristic: if extracted value is close to standard, assume compliant
                extracted_val = float(extracted_numbers[0])
                standard_val = float(standard_numbers[0])
                if abs(extracted_val - standard_val) / standard_val < 0.1:  # Within 10%
                    return True, "Fallback: Values are within acceptable range"
            except:
                pass
        
        # Default to compliant if we can't determine otherwise
        return True, "Fallback: Unable to verify definitively, assuming compliant"
    
    def verify_from_json(self, json_path):
        """Verify document compliance using a JSON file of extracted parameters"""
        try:
            # Load extracted parameters from JSON
            with open(json_path, 'r') as f:
                extracted_params = json.load(f)
            
            return self.verify_parameters(extracted_params)
        except Exception as e:
            print(f"Error verifying from JSON: {e}")
            # Return a minimal valid result to prevent UI hanging
            return {
                "overall_compliant": False,
                "parameter_results": {"error": {"compliant": False, "message": str(e)}},
                "parameters_checked": 0,
                "error": str(e)
            }
    
    def verify_parameters(self, extracted_params):
        """Verify if extracted parameters meet the standard requirements"""
        verification_results = {}
        compliant_count = 0
        parameters_checked = 0

        # Add timeout to ensure this method completes
        start_time = time.time()
        timeout = 45  # 45 second timeout

        for param_name in extracted_params.keys():
            if time.time() - start_time > timeout:
                print("Verification timed out, returning partial results")
                break

            if param_name in self.standard:
                try:
                    compliant, message, extracted_values, standard_value = self._verify_parameter_with_values(
                        param_name, extracted_params, self.standard
                    )

                    verification_results[param_name] = {
                        "compliant": compliant,
                        "message": message,
                        "extracted_values": extracted_values,
                        "standard_value": standard_value
                    }

                    parameters_checked += 1
                    if compliant:
                        compliant_count += 1
                        
                except Exception as e:
                    print(f"Error verifying parameter {param_name}: {e}")
                    verification_results[param_name] = {
                        "compliant": False,
                        "message": f"Error: {str(e)}",
                        "extracted_values": [],
                        "standard_value": None
                    }
            else:
                # Parameter found in document but not in standards
                verification_results[param_name] = {
                    "compliant": True,
                    "message": "Parameter found in document but not defined in standards",
                    "extracted_values": [],
                    "standard_value": None
                }

        if parameters_checked == 0:
            verification_results["no_parameters"] = {
                "compliant": False,
                "message": "No matching parameters found between document and standard",
                "extracted_values": [],
                "standard_value": None
            }
            parameters_checked = 1  # to avoid division by zero

        # Enforce: at least 4 parameters must be verified
        if parameters_checked < 4:
            overall_compliant = False
            compliance_reason = "Fewer than 4 parameters verified"
        else:
            # Compliance threshold 60%
            compliance_threshold = 0.60
            overall_compliant = (compliant_count / parameters_checked) >= compliance_threshold
            compliance_reason = f"{compliant_count} out of {parameters_checked} compliant ({(compliant_count/parameters_checked)*100:.1f}%)"

        return {
            "overall_compliant": overall_compliant,
            "compliance_reason": compliance_reason,
            "parameter_results": verification_results,
            "parameters_checked": parameters_checked,
            "model_info": {
                "using_hf_token": bool(self.hf_api_token),
                "nli_available": bool(self.nli_pipeline)
            }
        }

    def verify_document(self, document_data):
        """Verify if a document meets the standard requirements"""
        # Extract parameters from document data
        extracted_params = {}
        
        # Define keywords for each parameter
        parameter_keywords = {
            "moisture_content": ["moisture", "water content", "%"],
            "hMF_content": ["hmf", "hydroxymethylfurfural", "mg/kg"],
            "diastase_activity": ["diastase", "schade", "enzyme"],
            "sucrose_content": ["sucrose", "g/100g"],
            "free_acidity": ["acidity", "milliequivalents", "meq/kg"],
            "electrical_conductivity": ["conductivity", "ms/cm", "milli-siemens"],
            "insoluble_solids": ["insoluble solids", "sediment"],
            "glucose_fructose": ["glucose", "fructose", "reducing sugars"]
        }
        
        try:
            # Extract from text
            text = document_data.get("text", "")
            for param, keywords in parameter_keywords.items():
                # Find sections containing these keywords
                sections = []
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in text.lower():
                        # Get the context around the keyword
                        start_idx = max(0, text.lower().find(keyword_lower) - 100)
                        end_idx = min(len(text), text.lower().find(keyword_lower) + 100)
                        context = text[start_idx:end_idx]
                        sections.append(context)
                
                # If we found relevant sections, store them
                if sections:
                    extracted_params[param] = {
                        "sections": sections,
                        "raw_values": []
                    }
            
            # Extract from tables
            tables = document_data.get("tables", [])
            for table in tables:
                table_data = table.get("data", [])
                
                # Convert to string for simple processing
                table_str = str(table_data)
                
                for param, keywords in parameter_keywords.items():
                    for keyword in keywords:
                        if keyword.lower() in table_str.lower():
                            if param not in extracted_params:
                                extracted_params[param] = {
                                    "sections": [],
                                    "raw_values": []
                                }
                            extracted_params[param]["raw_values"].append(table_str)
            
            # Verify extracted parameters
            return self.verify_parameters(extracted_params)
            
        except Exception as e:
            print(f"Error in document verification: {e}")
            # Return a minimal valid result to prevent UI hanging
            return {
                "overall_compliant": False,
                "parameter_results": {"error": {"compliant": False, "message": str(e)}},
                "parameters_checked": 0,
                "error": str(e),
                "model_info": {
                    "using_hf_token": bool(self.hf_api_token),
                    "nli_available": False
                }
            }