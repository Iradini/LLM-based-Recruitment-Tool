import base64
from io import BytesIO

from langchain_core.messages import HumanMessage
from pypdf import PdfReader

from backend.llm_factory import get_llm


def extract_text_from_pdf(pdf_bytes: BytesIO) -> str:
    """
    Extract text from a PDF file.

    Parameters
    ----------
    pdf_bytes : BytesIO
        The PDF file as a BytesIO object.

    Returns
    -------
    pdf_text : str
        The extracted text from the PDF file.

    Raises
    ------
    ValueError
        If the provided PDF file is empty or invalid.

    Notes
    -----
    This function uses the pypdf library to extract text from a PDF file.
    It assumes that the PDF file is in a valid format and can be read.

    Examples
    --------
    >>> from io import BytesIO
    >>> pdf_bytes = BytesIO(b'%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 27 >>\nstream\nBT\n/F1 12 Tf\n100 100 Td\n(Hello, World!) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000103 00000 n\n0000000174 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n227\n%%EOF\n')
    >>> extract_text_from_pdf(pdf_bytes)
    'Hello, World!'
    """
    pdf_text = ""

    pdf = PdfReader(pdf_bytes)
    for page in pdf.pages:
        pdf_text += page.extract_text()

    return pdf_text


def pdf_to_markdown(pdf_bytes: BytesIO) -> str:
    """
    Convert a PDF to markdown using the LLM's vision capabilities.

    Sends the raw PDF as base64 to the multimodal LLM and asks it
    to faithfully transcribe the content as clean markdown.

    Parameters
    ----------
    pdf_bytes : BytesIO
        The PDF file as a BytesIO object.

    Returns
    -------
    markdown_text : str
        The PDF content as markdown.
    """
    # TODO: Read the raw bytes from the PDF and encode them as base64.
    # Hint: Use `pdf_bytes.read()` and `base64.standard_b64encode()`.

    # TODO: Get an LLM instance using the `get_llm` factory function.

    # TODO: Create a `HumanMessage` with multimodal content (text + media).
    # The message content should be a list with two items:
    # 1. A text block with instructions to convert the PDF to clean markdown.
    # 2. A media block with `mime_type: "application/pdf"` and the base64 data.
    # Hint: Use `{"type": "text", "text": "..."}` and
    #       `{"type": "media", "mime_type": "application/pdf", "data": b64}`.

    # TODO: Invoke the LLM with the message and return the response content.
    # Note: Gemini may return a list of content parts instead of a plain string.
    # If so, join the text parts together.
