"""
Shared embedding config — one place that initializes the HuggingFaceEmbeddings
object (model name, device config) used both to BUILD the FAISS index
(scripts/build_job_index.py) and to QUERY it (chains/gap_analysis_chain.py).

Keeping this in one file matters: if the build and query sides ever used
different embedding models, similarity search would silently return
garbage (vectors from two different models aren't comparable).
"""

from langchain_huggingface import HuggingFaceEmbeddings
from configs.settings import EMBEDDING_MODEL_NAME


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # switch to "cuda" in Colab if a GPU is available
    )
