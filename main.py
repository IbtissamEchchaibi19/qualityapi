import os
import json
import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pdf_processor import PDFProcessor
from verifier import DocumentVerifier
from certificate_gen import CertificateGenerator

app = FastAPI(title="Document Verification and Certification API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("standards", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("extracted_data", exist_ok=True)
os.makedirs("certificates", exist_ok=True)

# Components
processor = PDFProcessor(output_dir="extracted_data")
cert_generator = CertificateGenerator(output_dir="certificates")

# HuggingFace token - Try multiple environment variable names
hf_api_token = (
    os.getenv("HF_API_TOKEN") or 
    os.getenv("HUGGINGFACE_API_TOKEN") or 
    os.getenv("HF_TOKEN") or 
    os.getenv("HUGGINGFACE_TOKEN")
)

# Log token status (without revealing the actual token)
if hf_api_token:
    print(f"✅ HuggingFace API token found (length: {len(hf_api_token)})")
else:
    print("⚠️  No HuggingFace API token found - will use local model fallback")

# Define the only standard file to use
STANDARD_FILE = "standards/gso_honey_standard.json"


@app.get("/")
def root():
    """Health check endpoint"""
    token_status = "available" if hf_api_token else "not_available"
    return {
        "message": "Document Verification API is running", 
        "status": "healthy",
        "hf_token_status": token_status,
        "standard_file": STANDARD_FILE
    }


@app.get("/api/status")
def api_status():
    """Detailed API status"""
    return {
        "api_version": "1.0",
        "status": "healthy",
        "huggingface_token": "available" if hf_api_token else "not_available",
        "standard_file_exists": os.path.exists(STANDARD_FILE),
        "directories": {
            "standards": os.path.exists("standards"),
            "uploads": os.path.exists("uploads"),
            "extracted_data": os.path.exists("extracted_data"),
            "certificates": os.path.exists("certificates")
        }
    }


@app.post("/verify_document")
def verify_document(files: List[UploadFile] = File(...)):
    """Verify uploaded documents against standards"""
    if not os.path.exists(STANDARD_FILE):
        return JSONResponse(
            status_code=500, 
            content={"error": f"Standard file not found: {STANDARD_FILE}"}
        )

    results = []

    for file in files:
        try:
            pdf_bytes = file.file.read()
            
            # Check if PDF is scanned or digital
            is_scanned = processor.is_scanned_pdf(pdf_bytes)
            doc_type = "Scanned" if is_scanned else "Digital"

            # Extract data from PDF
            extracted_data = processor.process_pdf(pdf_bytes)

            # Save extracted parameters to JSON for debugging
            base_filename = os.path.splitext(file.filename)[0]
            param_json_path = os.path.join("extracted_data", f"{base_filename}_parameters.json")
            with open(param_json_path, "w") as f:
                json.dump(extracted_data.get("parameters", {}), f, indent=2)

            # Initialize verifier with HuggingFace token
            verifier = DocumentVerifier(STANDARD_FILE, hf_api_token=hf_api_token)
            
            # Verify document
            verification_results = verifier.verify_document(extracted_data)

            # Prepare response
            response = {
                "document": file.filename,
                "status": "passed" if verification_results["overall_compliant"] else "failed",
                "details": verification_results,
                "doc_type": doc_type,
                "processing_info": {
                    "parameter_file": param_json_path,
                    "model_info": verification_results.get("model_info", {})
                }
            }

            # Generate certificate if document passed
            if verification_results["overall_compliant"]:
                try:
                    cert_path = cert_generator.generate_certificate(
                        file.filename,
                        os.path.basename(STANDARD_FILE).replace(".json", ""),
                        verification_results
                    )
                    response["certificate_path"] = cert_path
                    response["certificate_generated"] = True
                except Exception as cert_error:
                    response["certificate_error"] = str(cert_error)
                    response["certificate_generated"] = False

            results.append(response)

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({
                "document": file.filename,
                "status": "error",
                "error": str(e)
            })

    return {
        "results": results,
        "processed_count": len(results),
        "api_info": {
            "hf_token_used": bool(hf_api_token),
            "standard_file": STANDARD_FILE
        }
    }


@app.post("/verify_from_json")
def verify_from_json(file: UploadFile = File(...)):
    """Verify document from pre-extracted JSON parameters"""
    if not os.path.exists(STANDARD_FILE):
        return JSONResponse(
            status_code=500, 
            content={"error": f"Standard file not found: {STANDARD_FILE}"}
        )
    
    try:
        # Read JSON file
        json_content = file.file.read()
        extracted_params = json.loads(json_content)
        
        # Initialize verifier
        verifier = DocumentVerifier(STANDARD_FILE, hf_api_token=hf_api_token)
        
        # Verify parameters
        verification_results = verifier.verify_parameters(extracted_params)
        
        response = {
            "document": file.filename,
            "status": "passed" if verification_results["overall_compliant"] else "failed",
            "details": verification_results,
            "source": "json_parameters"
        }
        
        return response
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing JSON file: {str(e)}"}
        )


@app.get("/list_certificates")
def list_certificates():
    """List all generated certificates"""
    certs = []
    if os.path.exists("certificates"):
        try:
            for f in os.listdir("certificates"):
                if f.endswith(".pdf"):
                    cert_path = os.path.join("certificates", f)
                    if os.path.exists(cert_path):
                        created = datetime.datetime.fromtimestamp(
                            os.path.getctime(cert_path)
                        ).isoformat()
                        file_size = os.path.getsize(cert_path)
                        certs.append({
                            "name": f,
                            "created_at": created,
                            "size_bytes": file_size
                        })
        except Exception as e:
            print(f"Error listing certificates: {e}")
    
    return {
        "certificates": certs,
        "total_count": len(certs)
    }


@app.get("/download_certificate/{filename}")
def download_certificate(filename: str):
    """Download a specific certificate"""
    path = os.path.join("certificates", filename)
    if os.path.exists(path):
        return FileResponse(
            path, 
            media_type="application/pdf", 
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        return JSONResponse(
            status_code=404, 
            content={"error": f"Certificate not found: {filename}"}
        )


@app.delete("/certificates/{filename}")
def delete_certificate(filename: str):
    """Delete a specific certificate"""
    path = os.path.join("certificates", filename)
    if os.path.exists(path):
        try:
            os.remove(path)
            return {"message": f"Certificate {filename} deleted successfully"}
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to delete certificate: {str(e)}"}
            )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": f"Certificate not found: {filename}"}
        )


# Run with uvicorn for deployment platforms like Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting server on {host}:{port}")
    print(f"🔑 HuggingFace token: {'✅ Available' if hf_api_token else '❌ Not found'}")
    print(f"📁 Standard file: {STANDARD_FILE} ({'✅ Exists' if os.path.exists(STANDARD_FILE) else '❌ Missing'})")
    
    uvicorn.run("main:app", host=host, port=port, reload=False)