import streamlit as st
import re
import numpy as np
import faiss

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Company Policy RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Company Policy RAG Assistant")

st.write(
    "Upload multiple company policy documents and ask questions. "
    "The assistant retrieves relevant information from all uploaded documents."
)


# ============================================================
# GROQ API KEY
# ============================================================

# PASTE YOUR NEW GROQ API KEY HERE
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

GROQ_MODEL = "openai/gpt-oss-20b"


if "PASTE_YOUR_API_KEY" in GROQ_API_KEY:

    st.warning(
        "Please enter your Groq API key in app.py"
    )

    st.stop()


client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:

    st.session_state.chunks = []


if "index" not in st.session_state:

    st.session_state.index = None


if "documents" not in st.session_state:

    st.session_state.documents = []


if "processed" not in st.session_state:

    st.session_state.processed = False


# IMPORTANT:
# This stores multiple questions and answers

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(filename, text):

    content = (
        filename.lower()
        + " "
        + text[:5000].lower()
    )

    if any(word in content for word in [
        "leave",
        "casual leave",
        "sick leave",
        "earned leave",
        "maternity leave"
    ]):

        return "Leave Policy"


    elif any(word in content for word in [
        "attendance",
        "working hours",
        "late attendance",
        "absence"
    ]):

        return "Attendance Policy"


    elif any(word in content for word in [
        "work from home",
        "remote work",
        "wfh",
        "hybrid"
    ]):

        return "Work From Home Policy"


    elif any(word in content for word in [
        "reimbursement",
        "travel expense",
        "expense claim",
        "medical expense"
    ]):

        return "Reimbursement Policy"


    elif any(word in content for word in [
        "information security",
        "confidential information",
        "data security",
        "it systems"
    ]):

        return "Information Security Policy"


    elif any(word in content for word in [
        "health and safety",
        "unsafe conditions",
        "workplace safety",
        "accidents"
    ]):

        return "Health & Safety Policy"


    elif any(word in content for word in [
        "environmental",
        "climate protection",
        "waste",
        "emissions"
    ]):

        return "Environmental Policy"


    elif any(word in content for word in [
        "sexual harassment",
        "discrimination",
        "harassment",
        "gender equality"
    ]):

        return "Harassment & Diversity Policy"


    elif any(word in content for word in [
        "employee",
        "human resource",
        "hr policy"
    ]):

        return "HR Policy"


    else:

        return "General Company Policy"


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file):

    reader = PdfReader(file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages.append({
                "text": text,
                "page": page_number
            })

    return pages


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file):

    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            text += paragraph.text + "\n"


    return [
        {
            "text": text,
            "page": "N/A"
        }
    ]


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(file):

    text = file.read().decode(
        "utf-8",
        errors="ignore"
    )

    return [
        {
            "text": text,
            "page": "N/A"
        }
    ]


# ============================================================
# DOCUMENT LOADER
# ============================================================

def load_document(file):

    filename = file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf(file)


    elif filename.endswith(".docx"):

        return extract_docx(file)


    elif filename.endswith(".txt"):

        return extract_txt(file)


    return []


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents(files):

    all_chunks = []

    document_names = []

    chunk_id = 0


    for file in files:

        filename = file.name

        document_names.append(filename)

        pages = load_document(file)


        # Detect category

        preview = ""

        for page in pages[:3]:

            preview += page["text"] + " "


        category = detect_category(
            filename,
            preview
        )


        file_type = filename.split(".")[-1].upper()


        # Process every page

        for page_data in pages:

            text = clean_text(
                page_data["text"]
            )


            if not text:

                continue


            chunks = create_chunks(text)


            for chunk in chunks:

                metadata = {

                    "source": filename,

                    "category": category,

                    "page": page_data["page"],

                    "file_type": file_type,

                    "chunk_id": chunk_id

                }


                all_chunks.append({

                    "text": chunk,

                    "metadata": metadata

                })


                chunk_id += 1


    return document_names, all_chunks


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_database(chunks):

    texts = [
        item["text"]
        for item in chunks
    ]


    embeddings = embedding_model.encode(

        texts,

        convert_to_numpy=True,

        normalize_embeddings=True

    )


    dimension = embeddings.shape[1]


    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        embeddings.astype(
            np.float32
        )
    )


    return index


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📁 Upload Documents")


    uploaded_files = st.file_uploader(

        "Upload company policy documents",

        type=[
            "pdf",
            "docx",
            "txt"
        ],

        accept_multiple_files=True

    )


    st.write(
        "You can upload multiple documents."
    )


    process_button = st.button(
        "🔄 Process Documents"
    )


    # Clear chat button

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if process_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one document."
        )


    else:

        with st.spinner(
            "Processing all documents..."
        ):

            try:

                document_names, chunks = process_documents(
                    uploaded_files
                )


                if not chunks:

                    st.error(
                        "No readable text found in documents."
                    )

                else:

                    index = create_vector_database(
                        chunks
                    )


                    st.session_state.documents = (
                        document_names
                    )

                    st.session_state.chunks = (
                        chunks
                    )

                    st.session_state.index = (
                        index
                    )

                    st.session_state.processed = True


                    # Clear previous chat when new
                    # documents are uploaded

                    st.session_state.chat_history = []


                    st.success(
                        f"Successfully processed "
                        f"{len(document_names)} documents."
                    )


                    st.info(
                        f"Created {len(chunks)} searchable chunks."
                    )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )


# ============================================================
# DISPLAY DOCUMENTS
# ============================================================

if st.session_state.processed:

    st.subheader("📄 Uploaded Documents")


    for document in st.session_state.documents:

        st.write(
            f"✅ {document}"
        )


    st.divider()


    st.subheader("🏷️ Document Categories")


    categories = set()


    for chunk in st.session_state.chunks:

        categories.add(
            chunk["metadata"]["category"]
        )


    for category in sorted(categories):

        st.write(
            f"• {category}"
        )


# ============================================================
# MULTI-DOCUMENT SEMANTIC SEARCH
# ============================================================

def search_documents(
    query,
    top_k=8
):

    # Convert question into embedding

    query_embedding = embedding_model.encode(

        [query],

        convert_to_numpy=True,

        normalize_embeddings=True

    )


    # Search ALL documents

    scores, indices = (
        st.session_state.index.search(

            query_embedding.astype(
                np.float32
            ),

            top_k * 3

        )
    )


    candidates = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:

            continue


        item = st.session_state.chunks[idx]


        candidates.append({

            "text": item["text"],

            "metadata": item["metadata"],

            "score": float(score)

        })


    # ========================================================
    # IMPORTANT:
    # Try to get information from DIFFERENT documents
    # ========================================================

    selected = []

    used_documents = set()


    # First pass:
    # one relevant chunk from each document

    for result in candidates:

        source = result["metadata"]["source"]


        if source not in used_documents:

            selected.append(result)

            used_documents.add(source)


    # Second pass:
    # add remaining highest scoring chunks

    for result in candidates:

        if len(selected) >= top_k:

            break


        if result not in selected:

            selected.append(result)


    return selected[:top_k]


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    question,
    results
):

    context = ""


    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]


        context += f"""

==================================================
SOURCE {i}
==================================================

Document: {metadata["source"]}

Category: {metadata["category"]}

Page: {metadata["page"]}

Similarity Score: {result["score"]:.4f}

Content:
{result["text"]}

"""


    system_prompt = """

You are a Company Policy RAG Assistant.

Your task is to answer employee questions using ONLY
the information provided in the retrieved company
policy documents.

IMPORTANT RULES:

1. Use only the supplied policy context.

2. Do NOT use outside knowledge.

3. Do NOT invent policies.

4. If information is available in multiple documents,
   combine the information from those documents.

5. Clearly distinguish which company/document provides
   each piece of information.

6. If the answer is not available in the documents,
   say:

   "I could not find this information in the uploaded
   company policy documents."

7. Do not make assumptions.

8. Preserve important dates, numbers, limits and
   conditions from the documents.

9. Always provide a Sources section.

10. Mention the document name and page number for
    relevant information.

"""


    user_prompt = f"""

RETRIEVED COMPANY POLICY CONTEXT:

{context}


EMPLOYEE QUESTION:

{question}


Please answer the question using the retrieved
company policy context.

If multiple documents are relevant, combine them.

At the end provide:

Sources:
- Document name - Page number
- Document name - Page number

"""


    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[

            {
                "role": "system",

                "content": system_prompt
            },

            {
                "role": "user",

                "content": user_prompt
            }

        ],

        temperature=0.1,

        max_tokens=1200

    )


    return response.choices[0].message.content


# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================

if st.session_state.chat_history:

    st.subheader("💬 Conversation")


    for chat in st.session_state.chat_history:

        # User question

        with st.chat_message("user"):

            st.write(
                chat["question"]
            )


        # Assistant answer

        with st.chat_message("assistant"):

            st.markdown(
                chat["answer"]
            )


            # Retrieved sources

            with st.expander(
                "📚 View Retrieved Sources"
            ):

                for i, result in enumerate(
                    chat["sources"],
                    start=1
                ):

                    metadata = result["metadata"]


                    st.markdown(
                        f"""
**Source {i}**

📄 **Document:** {metadata["source"]}

🏷️ **Category:** {metadata["category"]}

📑 **Page:** {metadata["page"]}

🔎 **Similarity:** {result["score"]:.4f}
"""
                    )


                    st.write(
                        result["text"]
                    )


                    st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the company policies..."
)


if question:

    if not st.session_state.processed:

        st.warning(
            "Please upload and process your documents first."
        )

    else:

        # ====================================================
        # RETRIEVE FROM ALL DOCUMENTS
        # ====================================================

        with st.spinner(
            "🔍 Searching all policy documents..."
        ):

            results = search_documents(
                question,
                top_k=8
            )


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        with st.spinner(
            "🤖 Generating policy-based answer..."
        ):

            answer = generate_answer(
                question,
                results
            )


        # ====================================================
        # SAVE CHAT
        # ====================================================

        st.session_state.chat_history.append({

            "question": question,

            "answer": answer,

            "sources": results

        })


        # Refresh page so chat appears

        st.rerun()