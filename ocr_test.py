# from PIL import Image
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# img = Image.open("Course Completion.png")
# text = pytesseract.image_to_string(img)
# print("---- OCR TEXT ----")
# print(text)



# ---------- ADVANCED OCR HELPERS (multi-format + PDF) ----------

# If you installed poppler, set the bin path here.
# If you don't want to use poppler yet, set POPPLER_PATH = None
POPPLER_PATH = r"C:\poppler\Library\bin"  # change if your poppler is elsewhere, or set to None

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "tiff", "bmp", "pdf"]


def extract_text_from_image(image):
    """Run Tesseract OCR on a single image."""
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(file_bytes):
    """
    1) Try PyPDF2 for normal text PDFs (most bank statements).
    2) If that fails or returns empty, fall back to pdf2image + Tesseract (needs Poppler).
    Returns: (text, preview_image or None)
    """
    # ---- 1) Try direct text extraction (no Poppler needed) ----
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            texts.append(f"\n\n===== PAGE {i} (text) =====\n\n{page_text}")
        full_text = "".join(texts).strip()
        if full_text:
            # We don't have a preview image here, but that's fine
            return full_text, None
    except Exception:
        # Ignore and fall back to image-based OCR
        pass

    # ---- 2) Fallback: convert pages to images + Tesseract ----
    kwargs = {}
    if POPPLER_PATH and os.path.isdir(POPPLER_PATH):
        kwargs["poppler_path"] = POPPLER_PATH

    try:
        pages = convert_from_bytes(file_bytes, dpi=300, **kwargs)
    except Exception as e:
        # Give a clearer error instead of "Unable to get page count"
        raise RuntimeError(
            f"PDF OCR failed. Likely Poppler is missing or POPPLER_PATH is wrong. Details: {e}"
        )

    all_text = []
    for i, page in enumerate(pages, start=1):
        page_text = pytesseract.image_to_string(page)
        all_text.append(f"\n\n===== PAGE {i} (image OCR) =====\n\n{page_text}")

    preview = pages[0] if pages else None
    return "\n".join(all_text), preview


def advanced_ocr(uploaded_file):
    """
    Detect file type (image / pdf) and route to correct OCR pipeline.
    Returns (text, preview_image)
    """
    file_bytes = uploaded_file.getvalue()
    filename = (uploaded_file.name or "").lower()

    is_pdf = filename.endswith(".pdf") or uploaded_file.type == "application/pdf"

    if is_pdf:
        return extract_text_from_pdf(file_bytes)
    else:
        image = Image.open(io.BytesIO(file_bytes))
        text = extract_text_from_image(image)
        return text, image
