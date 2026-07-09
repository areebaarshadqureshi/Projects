"""
app.py — Streamlit UI for Urdu Code-Switch Emotion Detection

Deployed on HuggingFace Spaces.
Loads fine-tuned XLM-RoBERTa model and provides interactive emotion prediction.
"""


import streamlit as st
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from predict import predict
from preprocess import clean_text, ID_TO_LABEL

# Emotion color palette for UI badges
EMOTION_COLORS = {
    "neutral": "#9E9E9E",   # Grey
    "anger": "#F44336",     # Red
    "fear": "#9C27B0",      # Purple
    "happy": "#4CAF50",     # Green
    "sad": "#2196F3",       # Blue
    "surprise": "#FF9800"   # Amber/Orange
}

# Emotion emoji mapping
EMOTION_EMOJIS = {
    "neutral": "😐",
    "anger": "😠",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲"
}

# Safety net: if these ever drift from config.py's canonical label set,
# fail loudly here at startup instead of a confusing KeyError mid-prediction.
assert set(EMOTION_COLORS) == set(ID_TO_LABEL.values()), \
    "EMOTION_COLORS is out of sync with config.ID2LABEL"
assert set(EMOTION_EMOJIS) == set(ID_TO_LABEL.values()), \
    "EMOTION_EMOJIS is out of sync with config.ID2LABEL"


@st.cache_resource
def load_model():
    """
    Load model and tokenizer from HuggingFace Hub.

    Cached to avoid reloading on every interaction.

    Returns:
        tuple: (model, tokenizer, loaded_ok). loaded_ok is False if the
        fine-tuned checkpoint couldn't be loaded — callers must check this
        and stop the page rather than serving predictions from an untrained
        fallback model (a randomly initialized head produces confident-
        looking but meaningless predictions).
    """
    model_name = "areebaarshad/urdu-emotion-xlmr"  # TODO: set to your real HF Hub repo

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model, tokenizer, True
    except Exception:
        return None, None, False


# Page config
st.set_page_config(
    page_title="Urdu Emotion Detector",
    page_icon="🎭",
    layout="centered"
)

# Header
st.title("🎭 Urdu Code-Switch Emotion Detector")
st.caption("Detects emotions in Roman Urdu–English code-switched text")

st.markdown("""
This app detects **6 emotions** in text written the way Pakistanis actually type on social media — 
mixing Roman Urdu and English in the same sentence.

**Emotions:** Neutral · Anger · Fear · Happy · Sad · Surprise
""")

# Load model
with st.spinner("Loading model..."):
    model, tokenizer, loaded_ok = load_model()

if not loaded_ok:
    st.error(
        "⚠️ Couldn't load the fine-tuned model from the Hub. The app can't "
        "make reliable predictions right now — please try refreshing in a "
        "minute. (If this persists, the Hub repo may be misconfigured.)"
    )
    st.stop()

st.success("Model loaded! Ready to detect emotions.")

# Input section
st.markdown("---")
st.subheader("Enter your text")

text = st.text_area(
    "Type or paste a sentence:",
    placeholder="yaar bohat bura hua 😢 I can't believe it",
    height=100,
    help="Mix Roman Urdu and English freely — that's the point!"
)

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Detect Emotion", use_container_width=True)

# Prediction
if predict_button:
    if not text.strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            cleaned_text = clean_text(text)
            try:
                # Use cleaned_text, not raw text — this is the same cleaning
                # applied at training time (preprocess.clean_text). Predicting
                # on raw text here would be train/serve skew.
                result = predict(cleaned_text, model, tokenizer)
            except Exception as e:
                st.error(f"Couldn't analyze that text: {e}")
                st.stop()

            predicted_label = result['label']
            confidence = result['confidence']
            all_scores = result['all_scores']

        # Display results
        st.markdown("---")
        st.subheader("Results")

        # Main prediction with colored badge
        color = EMOTION_COLORS[predicted_label]
        emoji = EMOTION_EMOJIS[predicted_label]

        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h1 style="color: white; margin: 0;">{emoji} {predicted_label.upper()}</h1>
                <p style="color: white; margin: 5px 0 0 0; font-size: 18px;">
                    Confidence: {confidence:.1%}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # All class scores
        st.markdown("### All Emotion Scores")

        # Sort scores for better visualization
        sorted_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

        df_scores = pd.DataFrame({
            'Emotion': list(sorted_scores.keys()),
            'Confidence': list(sorted_scores.values())
        })

        st.bar_chart(
            data=df_scores.set_index('Emotion'),
            height=300
        )

        # Show detailed scores
        with st.expander("View detailed scores"):
            for emotion, score in sorted_scores.items():
                emoji_icon = EMOTION_EMOJIS[emotion]
                st.write(f"{emoji_icon} **{emotion.capitalize()}**: {score:.4f} ({score*100:.2f}%)")

# Footer with disclaimer
st.markdown("---")
st.markdown("""
### ⚠️ Limitations

- **Model trained on ~20,000 sentences** from a research corpus
- **Fear and Surprise are less reliable** due to limited training data (~1.1% each)
- **Best for informal social media text** mixing Roman Urdu and English
- **Accuracy varies** — treat predictions as suggestions, not ground truth

### About

Built with:
- **Model:** XLM-RoBERTa-base fine-tuned for 6-class emotion classification
- **Dataset:** Roman Urdu-English Code-Switched Emotion Dataset
- **Framework:** HuggingFace Transformers + Streamlit

[📄 Project Repository](https://github.com/YOUR_USERNAME/urdu-emotion-detector) · 
[📊 Model Card](https://huggingface.co/YOUR_HF_USERNAME/urdu-emotion-xlmr)
""")
# TODO: replace YOUR_USERNAME / YOUR_HF_USERNAME above with your real GitHub
# and Hugging Face usernames before sharing this anywhere — the link in your
# last upload (.../areebaarshadqureshi/Projects/urdu-emotion-detector) has an
# extra "/Projects/" segment that doesn't match GitHub's repo URL shape
# (github.com/<user>/<repo>), so it's likely a dead link as-is. Also update
# model_name in load_model() above to match your real HF Hub repo.

st.caption("Made with ❤️ for the Pakistani NLP community")
