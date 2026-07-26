import json
from langsmith import traceable
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from configs.settings import DATA_DIR, VECTORSTORE_DIR
from embeddings.embedding_config import get_embeddings


@traceable(name="load_job_documents")
def load_job_documents() -> list[Document]:
    docs = []
    for file in (DATA_DIR / "job_requirements").glob("*.json"):
        postings = json.loads(file.read_text())
        for posting in postings:
            text = (
                f"Role: {posting['role']}\n"
                f"Seniority: {posting['seniority']}\n"
                f"Required skills: {', '.join(posting['required_skills'])}\n"
                f"Nice to have: {', '.join(posting.get('nice_to_have', []))}"
            )
            docs.append(Document(page_content=text, metadata={"role": posting["role"]}))
    return docs


def build_index():
    docs = load_job_documents()
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(VECTORSTORE_DIR / "job_requirements"))
    print(f"Indexed {len(docs)} job requirement documents")


if __name__ == "__main__":
    build_index()
