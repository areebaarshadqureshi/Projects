"""
configs/llm_client.py

Builds the LLM object actually used by the chains, based on values in
settings.py. Kept separate from settings.py on purpose: settings.py is
pure config (no API calls, no heavy imports), while this file does the
real work of instantiating a client.

Switches behavior based on settings.ENVIRONMENT:
  - "colab"      -> loads LOCAL_MODEL_ID directly via transformers.
                    Only viable with Colab's GPU -- do not use this branch
                    on a low-RAM laptop.
  - "production" -> calls HOSTED_MODEL_ID via Hugging Face's Inference
                    Providers. No local RAM/GPU needed, but every account
                    only gets a small monthly free credit allowance --
                    once exceeded, every call fails with a 402 Payment
                    Required, regardless of what the code does.
  - "groq"       -> calls GROQ_MODEL_ID via Groq's hosted API instead.
                    Still an open-source model (Llama 3.3), just served
                    through a different, faster host with a genuinely
                    usable free tier -- added specifically because the
                    "production" branch's free HF credit runs out fast
                    for a project that gets used/tested this often.
"""

import os
from dotenv import load_dotenv
from configs.settings import ENVIRONMENT, LOCAL_MODEL_ID, HOSTED_MODEL_ID, GROQ_MODEL_ID

load_dotenv()


def get_llm():
    if ENVIRONMENT == "colab":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
        return HuggingFacePipeline(pipeline=pipe)

    elif ENVIRONMENT == "production":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        hf_token = os.getenv("HF_API_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_API_TOKEN not found. Copy .env.example to .env "
                "and fill in your Hugging Face token."
            )
        endpoint = HuggingFaceEndpoint(
            repo_id=HOSTED_MODEL_ID,
            huggingfacehub_api_token=hf_token,
            task="text-generation",
            max_new_tokens=1024,
            temperature=0.1,  # low temp -- used for structured extraction, not creative output
        )
        return ChatHuggingFace(llm=endpoint)

    elif ENVIRONMENT == "groq":
        from langchain_groq import ChatGroq

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError(
                "GROQ_API_KEY not found. Copy .env.example to .env "
                "and fill in your Groq API key (console.groq.com)."
            )
        return ChatGroq(
            model=GROQ_MODEL_ID,
            api_key=groq_key,
            max_tokens=1024,
            temperature=0.1,  # low temp -- used for structured extraction, not creative output
            max_retries=5,    # default is 2 -- bumped up since concurrent .batch() calls
                               # (repo audits, contribution filtering) can burst past
                               # Groq's free-tier rate limit; each retry backs off
                               # automatically (built into the groq SDK's client)
        )

    else:
        raise ValueError(
            f"Unknown ENVIRONMENT: {ENVIRONMENT!r} in settings.py "
            "(expected 'colab', 'production', or 'groq')"
        )
