import os
from datetime import datetime
from fpdf import FPDF
import uuid

class CertificateGenerator:
    def __init__(self, output_dir="certificates"):
        """Initialize the certificate generator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_certificate(self, document_name, standard_name, verification_results):
        """Generate a simple one-page PDF certificate for a compliant document"""
        if not verification_results.get("overall_compliant", False):
            return None
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=False)  # Disable automatic page breaks

        # Colors
        gold_rgb = (218, 165, 32)
        dark_blue_rgb = (25, 25, 112)
        gray_rgb = (100, 100, 100)
        green_rgb = (0, 128, 0)

        # Start at a controlled position
        pdf.set_y(30)

        # Title Section
        pdf.set_font("Arial", "B", 28)  # Reduced font size
        pdf.set_text_color(*gold_rgb)
        pdf.cell(0, 15, "CERTIFICATE OF QUALITY", 0, 1, "C")
        pdf.ln(8)

        # Subtitle
        pdf.set_font("Arial", "", 14)
        pdf.set_text_color(*dark_blue_rgb)
        pdf.cell(0, 10, "This is to certify that", 0, 1, "C")
        pdf.ln(3)

        # Document name in larger font
        pdf.set_font("Arial", "B", 18)  # Reduced font size
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 12, f'"{document_name}"', 0, 1, "C")
        pdf.ln(8)

        # Main certification statement
        pdf.set_font("Arial", "", 14)
        pdf.set_text_color(*green_rgb)
        pdf.cell(0, 10, "HAS SUCCESSFULLY PASSED", 0, 1, "C")
        pdf.cell(0, 10, "ALL QUALITY VERIFICATION STANDARDS", 0, 1, "C")
        pdf.ln(12)

        # Standard applied
        pdf.set_font("Arial", "", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, f"Standards Applied: {standard_name}", 0, 1, "C")
        pdf.ln(8)

        # Parameters passed (simple count)
        parameters_passed = len([r for r in verification_results.get("parameter_results", {}).values() 
                               if r.get("compliant", False)])
        total_parameters = len(verification_results.get("parameter_results", {}))
        
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(*green_rgb)
        pdf.cell(0, 8, f"Parameters Verified: {parameters_passed}/{total_parameters} PASSED", 0, 1, "C")
        pdf.ln(15)

        # Certificate details - simplified without rectangle
        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(*gray_rgb)
        
        # Certificate info without box
        cert_id = f"CERT-{uuid.uuid4().hex[:8].upper()}"
        current_date = datetime.now().strftime('%B %d, %Y')
        
        pdf.cell(0, 6, f"Certificate ID: {cert_id}", 0, 1, "C")
        pdf.cell(0, 6, f"Issue Date: {current_date}", 0, 1, "C")
        pdf.cell(0, 6, f"Valid From: {current_date}", 0, 1, "C")
        pdf.cell(0, 6, "Status: COMPLIANT", 0, 1, "C")
        
        pdf.ln(15)

        # Signature area
        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, "Authorized by:", 0, 1, "C")
        pdf.ln(8)
        
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, "Fujairah Research Center", 0, 1, "C")
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(*gray_rgb)

        # Footer - using current position instead of absolute positioning
        pdf.set_text_color(*gray_rgb)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 5, "This certificate confirms compliance with specified quality standards.", 0, 1, "C")

        # Save
        filename = f"{self.output_dir}/{document_name.replace(' ', '_')}_Certificate_{datetime.now().strftime('%Y%m%d')}.pdf"
        pdf.output(filename)
        return filename

if __name__ == "__main__":
    generator = CertificateGenerator()

    sample_results = {
        "overall_compliant": True,
        "parameter_results": {
            "moisture_content": {"compliant": True, "message": "Confidence: 0.85"},
            "hMF_content": {"compliant": True, "message": "Confidence: 0.92"},
            "diastase_activity": {"compliant": True, "message": "Confidence: 0.78"},
            "sucrose_content": {"compliant": True, "message": "Confidence: 0.88"},
            "free_acidity": {"compliant": True, "message": "Confidence: 0.91"},
            "electrical_conductivity": {"compliant": True, "message": "Confidence: 0.87"},
        }
    }

    certificate_path = generator.generate_certificate(
        "LauraHoney4.pdf", 
        "gso_honey_standard",
        sample_results
    )

    print(f"Certificate generated: {certificate_path}")