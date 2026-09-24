# ============================================================
# INSTALLATION
# ============================================================
#
# pip install pymupdf4llm
# pip install sentence-transformers
# pip install faiss-cpu
# pip install transformers
# pip install accelerate
# pip install gradio
#
# ============================================================


import os
import re
import numpy as np
import faiss
import gradio as gr
import torch

import pymupdf4llm

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. PDF PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FILE = os.path.join(
    BASE_DIR,
    "Earth Our Planet data from web.pdf"
)

if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(
        f"PDF not found: {PDF_FILE}"
    )


# ============================================================
# 2. PDF → MARKDOWN
# ============================================================

text = pymupdf4llm.to_markdown(PDF_FILE)

print("Extracted characters:", len(text))


# ============================================================
# 3. STRUCTURE-AWARE CHUNKING
# ============================================================

def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 1200,
    overlap: int = 200
):

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    text = text.strip()

    if not text:
        raise ValueError(
            "No text was extracted from the PDF."
        )

    # --------------------------------------------------------
    # Markdown headings
    # --------------------------------------------------------

    lines = text.split("\n")

    current_chapter = None
    current_section = None
    current_subsection = None

    sections = []
    current_content = []

    # --------------------------------------------------------
    # Save section
    # --------------------------------------------------------

    def save_section():

        if current_content:

            content = "\n".join(
                current_content
            ).strip()

            if content:

                sections.append({
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection,
                    "text": content
                })

    # --------------------------------------------------------
    # Build hierarchical sections
    # --------------------------------------------------------

    for line in lines:

        line = line.strip()

        if not line:

            if (
                current_content
                and current_content[-1] != ""
            ):
                current_content.append("")

            continue

        # H1
        if re.match(r"^#\s+", line):

            save_section()

            current_content = []

            current_chapter = re.sub(
                r"^#\s+",
                "",
                line
            )

            current_section = None
            current_subsection = None

            current_content.append(line)

        # H2
        elif re.match(r"^##\s+", line):

            save_section()

            current_content = []

            current_section = re.sub(
                r"^##\s+",
                "",
                line
            )

            current_subsection = None

            current_content.append(line)

        # H3
        elif re.match(r"^###\s+", line):

            save_section()

            current_content = []

            current_subsection = re.sub(
                r"^###\s+",
                "",
                line
            )

            current_content.append(line)

        else:

            current_content.append(line)

    save_section()

    # ========================================================
    # Split sections into chunks
    # ========================================================

    documents = []

    chunk_counter = 0

    for section in sections:

        section_text = section["text"]

        # ----------------------------------------------------
        # Small section
        # ----------------------------------------------------

        if len(section_text) <= chunk_size:

            documents.append({

                "id": f"chunk_{chunk_counter}",

                "source": source,

                "chunk_index": chunk_counter,

                "chapter": section["chapter"],

                "section": section["section"],

                "subsection": section["subsection"],

                "text": section_text

            })

            chunk_counter += 1

            continue

        # ----------------------------------------------------
        # Large section
        # ----------------------------------------------------

        paragraphs = re.split(
            r"\n\s*\n",
            section_text
        )

        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # ------------------------------------------------
            # Paragraph fits
            # ------------------------------------------------

            if (
                len(current_chunk)
                + len(paragraph)
                + 2
                <= chunk_size
            ):

                if current_chunk:
                    current_chunk += "\n\n"

                current_chunk += paragraph

            # ------------------------------------------------
            # Paragraph does not fit
            # ------------------------------------------------

            else:

                if current_chunk:

                    documents.append({

                        "id": f"chunk_{chunk_counter}",

                        "source": source,

                        "chunk_index": chunk_counter,

                        "chapter": section["chapter"],

                        "section": section["section"],

                        "subsection": section["subsection"],

                        "text": current_chunk.strip()

                    })

                    chunk_counter += 1

                # ------------------------------------------------
                # Very large paragraph
                # ------------------------------------------------

                if len(paragraph) > chunk_size:

                    start = 0

                    while start < len(paragraph):

                        end = start + chunk_size

                        small_chunk = (
                            paragraph[start:end]
                            .strip()
                        )

                        if small_chunk:

                            documents.append({

                                "id": (
                                    f"chunk_{chunk_counter}"
                                ),

                                "source": source,

                                "chunk_index":
                                    chunk_counter,

                                "chapter":
                                    section["chapter"],

                                "section":
                                    section["section"],

                                "subsection":
                                    section["subsection"],

                                "text":
                                    small_chunk

                            })

                            chunk_counter += 1

                        start = end - overlap

                    current_chunk = ""

                else:

                    current_chunk = paragraph

        # ----------------------------------------------------
        # Save final chunk
        # ----------------------------------------------------

        if current_chunk:

            documents.append({

                "id": f"chunk_{chunk_counter}",

                "source": source,

                "chunk_index": chunk_counter,

                "chapter": section["chapter"],

                "section": section["section"],

                "subsection": section["subsection"],

                "text": current_chunk.strip()

            })

            chunk_counter += 1

    # ========================================================
    # Previous / next relationships
    # ========================================================

    for i, document in enumerate(documents):

        document["previous_chunk"] = (
            documents[i - 1]["id"]
            if i > 0
            else None
        )

        document["next_chunk"] = (
            documents[i + 1]["id"]
            if i < len(documents) - 1
            else None
        )

    return documents


# ============================================================
# CREATE DOCUMENT CHUNKS
# ============================================================

documents = chunk_document(
    text=text,
    source="Earth Our Planet data from web.pdf",
    chunk_size=1200,
    overlap=200
)

print(
    "Number of chunks:",
    len(documents)
)


# ============================================================
# 4. EMBEDDING MODEL
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Embedding device:", device)


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)


# ============================================================
# 5. GENERATE DOCUMENT EMBEDDINGS
# ============================================================

texts = [
    document["text"]
    for document in documents
]


embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


embedding_matrix = np.asarray(
    embeddings
).astype("float32")


print(
    "Embedding shape:",
    embedding_matrix.shape
)


# ============================================================
# 6. FAISS INDEX
# ============================================================

dimension = embedding_matrix.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embedding_matrix
)


print(
    "FAISS vectors:",
    index.ntotal
)

print(
    "Embedding dimension:",
    dimension
)


# ============================================================
# 7. SEMANTIC SEARCH
# ============================================================

def search_documents(
    query,
    top_k=5
):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        result = documents[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# 8. BUILD RAG CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
--- DOCUMENT CHUNK {result['chunk_index']} ---

Chapter:
{result['chapter']}

Section:
{result['section']}

Subsection:
{result['subsection']}

Content:
{result['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# 9. LOAD GENERATIVE AI MODEL
# ============================================================

# Small model for demonstration.
# You can replace this with another compatible
# Hugging Face causal language model.

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    LLM_NAME
)


# -----------------------------------------------------
# LOAD LLM
# -----------------------------------------------------

if torch.cuda.is_available():

    print("LLM device: CUDA")

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        dtype=torch.float16,
        device_map="auto"
    )

else:

    print("LLM device: CPU")

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        dtype=torch.float32
    )

llm.eval()

print("LLM loaded successfully")


# ============================================================
# 10. GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context
):

    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information
contained in the DOCUMENT CONTEXT.

Rules:
1. Do not use outside knowledge.
2. Give a concise and direct answer.
3. Do not add unrelated information.
4. Do not invent facts.
5. If the answer is not present in the context,
   say exactly:
   "I could not find the answer in the document."

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(llm.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = llm.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True
        )

    generated_tokens = output[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# 11. COMPLETE RAG PIPELINE
# ============================================================

def rag_pipeline(
    question,
    top_k=3
):

    # ------------------------------------------
    # RETRIEVAL
    # ------------------------------------------

    results = search_documents(
        question,
        top_k=top_k
    )

    if not results:

        return (
            "No relevant information was found.",
            [],
            ""
        )

    # ------------------------------------------
    # CONTEXT
    # ------------------------------------------

    context = build_context(
        results
    )

    # ------------------------------------------
    # GENERATION
    # ------------------------------------------

    answer = generate_answer(
        question,
        context
    )

    # ------------------------------------------
    # SOURCES
    # ------------------------------------------

    sources = []

    for result in results:

        sources.append({

            "chunk": result[
                "chunk_index"
            ],

            "score": round(
                result["score"],
                4
            ),

            "chapter":
                result["chapter"],

            "section":
                result["section"]

        })

    return (
        answer,
        sources,
        context
    )


# ============================================================
# 12. RETRIEVAL EVALUATION
# ============================================================

def hit_at_k(
    retrieved_chunks,
    relevant_chunks,
    k
):

    retrieved = retrieved_chunks[:k]

    return int(
        any(
            chunk in relevant_chunks
            for chunk in retrieved
        )
    )


def precision_at_k(
    retrieved_chunks,
    relevant_chunks,
    k
):

    retrieved = retrieved_chunks[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        chunk in relevant_chunks
        for chunk in retrieved
    )

    return relevant_count / len(
        retrieved
    )


def recall_at_k(
    retrieved_chunks,
    relevant_chunks,
    k
):

    if not relevant_chunks:
        return 0.0

    retrieved = retrieved_chunks[:k]

    relevant_count = sum(
        chunk in relevant_chunks
        for chunk in retrieved
    )

    return relevant_count / len(
        relevant_chunks
    )


def reciprocal_rank(
    retrieved_chunks,
    relevant_chunks
):

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        if chunk in relevant_chunks:

            return 1 / rank

    return 0.0


# ============================================================
# 13. EVALUATION DATASET
# ============================================================

# IMPORTANT:
# These relevant chunk IDs are examples.
#
# You must replace them with the actual relevant
# chunk indexes from your PDF.

evaluation_data = [

    {
        "question":
            "What are the main features of Earth?",

        "relevant_chunks":
            [5, 8]
    },

    {
        "question":
            "What is Earth made of?",

        "relevant_chunks":
            [10, 11]
    },

    {
        "question":
            "What is Earth's atmosphere?",

        "relevant_chunks":
            [15, 16]
    }
]


# ============================================================
# 14. RUN RETRIEVAL EVALUATION
# ============================================================

def evaluate_retrieval(
    evaluation_data,
    k=5
):

    hit_scores = []
    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    for item in evaluation_data:

        results = search_documents(
            item["question"],
            top_k=k
        )

        retrieved_chunks = [
            result["chunk_index"]
            for result in results
        ]

        relevant_chunks = set(
            item["relevant_chunks"]
        )

        hit_scores.append(
            hit_at_k(
                retrieved_chunks,
                relevant_chunks,
                k
            )
        )

        precision_scores.append(
            precision_at_k(
                retrieved_chunks,
                relevant_chunks,
                k
            )
        )

        recall_scores.append(
            recall_at_k(
                retrieved_chunks,
                relevant_chunks,
                k
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved_chunks,
                relevant_chunks
            )
        )

    metrics = {

        f"Hit@{k}":
            np.mean(hit_scores),

        f"Precision@{k}":
            np.mean(precision_scores),

        f"Recall@{k}":
            np.mean(recall_scores),

        "MRR":
            np.mean(reciprocal_ranks)

    }

    return metrics


# ============================================================
# 15. RUN EVALUATION
# ============================================================

retrieval_metrics = evaluate_retrieval(
    evaluation_data,
    k=5
)


print("\n")
print("=" * 60)
print("RETRIEVAL EVALUATION")
print("=" * 60)

for metric, value in retrieval_metrics.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ============================================================
# 16. GENERATION EVALUATION DATA
# ============================================================

generation_test_data = [

    {
        "question":
            "What are the main features of Earth?",

        "ground_truth":
            "Earth has an atmosphere, oceans, continents, and a solid surface."
    },

    {
        "question":
            "What is Earth made of?",

        "ground_truth":
            "Earth consists of different layers and materials that make up its interior and surface."
    }
]

# ============================================================
# 17. GENERATE TEST ANSWERS
# ============================================================

generated_answers = []


for item in generation_test_data:

    answer, sources, context = rag_pipeline(
        item["question"],
        top_k=3
    )

    generated_answers.append({

        "question":
            item["question"],

        "ground_truth":
            item["ground_truth"],

        "generated_answer":
            answer,

        "context":
            context,

        "sources":
            sources

    })


# ============================================================
# 18. DISPLAY GENERATION RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("GENERATION RESULTS")
print("=" * 60)


for item in generated_answers:

    print("\nQUESTION:")
    print(item["question"])

    print("\nGROUND TRUTH:")
    print(item["ground_truth"])

    print("\nGENERATED ANSWER:")
    print(item["generated_answer"])

    print("\nSOURCES:")
    print(item["sources"])

    print("-" * 60)


# ============================================================
# 19. GRADIO CHATBOT
# ============================================================

def chatbot(
    message,
    history
):

    try:

        answer, sources, context = rag_pipeline(
            message,
            top_k=3
        )

        response = (
            "### Answer\n\n"
            f"{answer}\n\n"
        )

        response += (
            "### Sources\n\n"
        )

        for source in sources:

            response += (
                f"- Chunk: "
                f"{source['chunk']}\n"
                f"- Similarity: "
                f"{source['score']}\n"
                f"- Chapter: "
                f"{source['chapter']}\n"
                f"- Section: "
                f"{source['section']}\n\n"
            )

        return response

    except Exception as e:

        return (
            f"Error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# 20. GRADIO INTERFACE
# ============================================================

demo = gr.ChatInterface(

    fn=chatbot,

    title=(
        "AI Document Intelligence - "
        "RAG Application"
    ),

    description=(
        "Ask questions about the Earth document. "
        "The system retrieves relevant document "
        "chunks using FAISS and generates answers "
        "using a language model."
    )
)


# ============================================================
# 21. LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch()
