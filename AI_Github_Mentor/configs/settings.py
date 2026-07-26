from pathlib import Path

SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
OUTPUT_DIR = BASE_DIR / "output"

GITHUB_API_BASE = "https://api.github.com"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# switch this to "colab" for local dev, "production" for hosted HF
# Inference Providers, or "groq" for Groq's hosted API (open-source
# models, generous free tier -- default here since HF's free monthly
# Inference Providers credit runs out fast for a project used this often)
ENVIRONMENT = "groq"

LOCAL_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HOSTED_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
GROQ_MODEL_ID = "llama-3.1-8b-instant"  

TOP_K_RETRIEVAL = 5

for folder in [DATA_DIR, VECTORSTORE_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
