from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from config import GOOGLE_API_KEY

# ── Prompt template ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a scientific research assistant. Answer the question using ONLY the research paper excerpts provided below.
If the answer is not covered in the excerpts, say: "I cannot find this in the provided papers."
Be concise, accurate, and cite which excerpt supports your answer where possible."""


def get_answer(user_query: str, db, top_k: int = 5) -> tuple[str, list]:
    """
    Retrieve relevant chunks from FAISS, then ask Gemini to answer based on them.

    Returns:
        answer  (str)  — LLM-generated response
        sources (list) — list of LangChain Document objects used as context
    """
    if not GOOGLE_API_KEY:
        return "⚠️ GOOGLE_API_KEY is not set. Add it to app/.env.", []

    # 1. Retrieve top-k relevant chunks
    sources = db.similarity_search(user_query, k=top_k)

    # 2. Build context block
    context_parts = []
    for i, doc in enumerate(sources, start=1):
        paper = doc.metadata.get("source", "unknown").replace(".pdf.txt", "")
        context_parts.append(f"[Excerpt {i} — {paper}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # 3. Build prompt
    prompt = f"""{SYSTEM_PROMPT}

---
{context}
---

Question: {user_query}

Answer:"""

    # 4. Call Gemini Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content, sources
