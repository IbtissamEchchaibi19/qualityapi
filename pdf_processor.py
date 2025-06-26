import os
import json
import re
import io
import fitz                 # PyMuPDF
import PyPDF2
import camelot
import base64
from PIL import Image
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.exceptions import HttpResponseError

class PDFProcessor:
    # ---------- initialisation ------------------------------------------------
    def __init__(self, output_dir: str = "extracted_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # ---- load Azure credentials ----
        load_dotenv()
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if key:
            print(f"Azure Key loaded: {key[:4]}…{key[-4:]}  |  Endpoint: {endpoint}")
        else:
            raise EnvironmentError(
                "Azure credentials not found – add AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT "
                "and AZURE_DOCUMENT_INTELLIGENCE_KEY to your .env file."
            )

        self.document_client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        # ---- parameters and their keyword triggers ----
        self.parameter_keywords = {
            "moisture_content":         ["moisture", "water content", "humidity", "%"],
            "hMF_content":              ["hmf", "hydroxymethylfurfural", "mg/kg"],
            "diastase_activity":        ["diastase", "schade", "enzyme"],
            "sucrose_content":          ["sucrose", "g/100g"],
            "free_acidity":             ["acidity", "milliequivalents", "meq/kg"],
            "electrical_conductivity":  ["conductivity", "ms/cm", "milli-siemens", "µmhos/cm"],
            "insoluble_solids":         ["insoluble solids", "sediment"],
            "glucose_fructose":         ["glucose", "fructose", "reducing sugars"],
        }

        # Pre-compile a regex for numeric value extraction
        units = r"%|mg\/kg|g\/100g|meq\/kg|ms\/cm"
        self.value_re = re.compile(rf"(\d+\.?\d*)\s*({units})", re.IGNORECASE)

    # ---------- small helpers -------------------------------------------------
    def _contains_keyword(self, text: str, param: str) -> bool:
        """Case-insensitive check if any keyword for *param* occurs in *text*."""
        lower = text.lower()
        return any(k in lower for k in self.parameter_keywords[param])

    def _all_params_found(self, found: set) -> bool:
        return len(found) == len(self.parameter_keywords)

    # ---------- PDF type check ------------------------------------------------
    def is_scanned_pdf(self, pdf_bytes: bytes) -> bool:
        """True if first page yields <100 characters – a heuristic for scans."""
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            reader = PyPDF2.PdfReader(pdf_stream)
            sample = reader.pages[0].extract_text() or ""
            return len(sample.strip()) < 100
        except Exception:
            # On any error, default to 'scanned'
            return True

    # ---------- Azure layout call --------------------------------------------
    def analyze_document(self, blob: bytes):
        try:
            base64_data = base64.b64encode(blob).decode("utf-8")
            poller = self.document_client.begin_analyze_document(
            model_id="prebuilt-layout",
            body={"base64Source": base64_data}
            )
            return poller.result()
        except HttpResponseError as e:
            print("Azure error:", e)
        except Exception as e:
            print("General analysis error:", e)
        return None
    


    # ---------- scanned PDFs (OCR path) ---------------------------------------
    def _pdf_to_images(self, pdf_bytes: bytes):
        images = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            zoom = fitz.Matrix(300 / 72, 300 / 72)
            for page in doc:
                pix = page.get_pixmap(matrix=zoom)
                images.append(Image.open(io.BytesIO(pix.tobytes("jpeg"))))
        finally:
            doc.close()
        return images

    def _extract_from_scanned(self, pdf_bytes: bytes):
        out = {"text": "", "tables": []}
        found_params = set()

        for img in self._pdf_to_images(pdf_bytes):
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            result = self.analyze_document(buf.getvalue())
            if not result:
                continue

            # ---- text lines ----
            for page in getattr(result, "pages", []):
                for ln in getattr(page, "lines", []):
                    for param in self.parameter_keywords:
                        if (
                            param not in found_params
                            and self._contains_keyword(ln.content, param)
                        ):
                            out["text"] += ln.content + "\n"
                            found_params.add(param)
                    if self._all_params_found(found_params):
                        break
                if self._all_params_found(found_params):
                    break
            if self._all_params_found(found_params):
                break

            # ---- tables ----
            for tb in getattr(result, "tables", []):
                keep = False
                for cell in tb.cells:
                    for param in self.parameter_keywords:
                        if (
                            param not in found_params
                            and self._contains_keyword(cell.content, param)
                        ):
                            keep = True
                            found_params.add(param)
                if keep:
                    table_data = [["" for _ in range(tb.column_count)] for _ in range(tb.row_count)]
                    for cell in tb.cells:
                        table_data[cell.row_index][cell.column_index] = cell.content
                    out["tables"].append({"table_id": f"table{len(out['tables'])+1}", "data": table_data})
                if self._all_params_found(found_params):
                    break
            if self._all_params_found(found_params):
                break
        return out

    # ---------- born-digital PDFs --------------------------------------------
    def _extract_from_digital(self, pdf_bytes: bytes):
        out = {"text": "", "tables": []}
        found_params = set()

        result = self.analyze_document(pdf_bytes)
        if result:  # Azure succeeded
            # ---- text ----
            for page in getattr(result, "pages", []):
                for ln in getattr(page, "lines", []):
                    for param in self.parameter_keywords:
                        if (
                            param not in found_params
                            and self._contains_keyword(ln.content, param)
                        ):
                            out["text"] += ln.content + "\n"
                            found_params.add(param)
                    if self._all_params_found(found_params):
                        break
                if self._all_params_found(found_params):
                    break

            # ---- tables ----
            for tb in getattr(result, "tables", []):
                keep = False
                for cell in tb.cells:
                    for param in self.parameter_keywords:
                        if (
                            param not in found_params
                            and self._contains_keyword(cell.content, param)
                        ):
                            keep = True
                            found_params.add(param)
                if keep:
                    table_data = [["" for _ in range(tb.column_count)] for _ in range(tb.row_count)]
                    for cell in tb.cells:
                        table_data[cell.row_index][cell.column_index] = cell.content
                    out["tables"].append({"table_id": f"table{len(out['tables'])+1}", "data": table_data})
                if self._all_params_found(found_params):
                    break

            return out

        # ---- fallback: PyPDF2 & Camelot ----
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                text = page.extract_text() or ""
                for param in self.parameter_keywords:
                    if (
                        param not in found_params
                        and self._contains_keyword(text, param)
                    ):
                        out["text"] += text + "\n"
                        found_params.add(param)
                if self._all_params_found(found_params):
                    break
        except Exception as e:
            print("PyPDF2 text extraction error:", e)

        try:
            with open("tmp.pdf", "wb") as fh:
                fh.write(pdf_bytes)
            tables = camelot.read_pdf("tmp.pdf", flavor="stream", pages="all")
            os.remove("tmp.pdf")
            for i, tbl in enumerate(tables):
                tbl_str = tbl.df.to_string().lower()
                keep = False
                for param in self.parameter_keywords:
                    if (
                        param not in found_params
                        and any(k in tbl_str for k in self.parameter_keywords[param])
                    ):
                        keep = True
                        found_params.add(param)
                if keep:
                    out["tables"].append(
                        {"table_id": f"table{i+1}", "data": tbl.df.values.tolist()}
                    )
                if self._all_params_found(found_params):
                    break
        except Exception as e:
            print("Camelot table error:", e)
            if os.path.exists("tmp.pdf"):
                os.remove("tmp.pdf")

        return out

    # ---------- parameter post-processing -------------------------------------
    def _extract_parameters(self, doc_data: dict):
        params = {}
        txt = doc_data.get("text", "").lower()

        # ---- from text context ----
        for param in self.parameter_keywords:
            if any(k in txt for k in self.parameter_keywords[param]):
                params.setdefault(param, {"contexts": [], "raw_values": []})
                # capture ±100 chars around each hit
                for m in re.finditer("|".join(self.parameter_keywords[param]), txt):
                    start = max(m.start() - 100, 0)
                    end = min(m.end() + 100, len(txt))
                    params[param]["contexts"].append(txt[start:end])

        # ---- from tables ----
        for tbl in doc_data.get("tables", []):
            tbl_str = str(tbl["data"]).lower()
            for param in self.parameter_keywords:
                if any(k in tbl_str for k in self.parameter_keywords[param]):
                    params.setdefault(param, {"contexts": [], "raw_values": []})
                    params[param]["raw_values"].append(tbl["data"])

        # ---- numeric value extraction ----
        for param, dat in params.items():
            vals = []
            for raw in dat.get("raw_values", []) + dat.get("contexts", []):
                for val, unit in self.value_re.findall(str(raw)):
                    vals.append(f"{val} {unit}")
            if vals:
                dat["values"] = vals

        return params

    # ---------- public entry-point -------------------------------------------
    def process_pdf(self, src):
        """*src* can be a file path or **bytes**."""
        if isinstance(src, str):
            if not os.path.exists(src):
                return {"error": f"{src} not found"}
            with open(src, "rb") as fh:
                pdf_bytes = fh.read()
        else:
            pdf_bytes = src  # already bytes

        print("Detecting PDF type…")
        scanned = self.is_scanned_pdf(pdf_bytes)
        print("  ➜ scanned" if scanned else "  ➜ born-digital")

        doc_data = (
            self._extract_from_scanned(pdf_bytes)
            if scanned
            else self._extract_from_digital(pdf_bytes)
        )

        if "error" in doc_data:
            return doc_data

        doc_data["parameters"] = self._extract_parameters(doc_data)
        return doc_data