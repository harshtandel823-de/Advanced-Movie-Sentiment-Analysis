
from pathlib import Path
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_sentiment_model"

MODEL_RESULTS = [
    {"Rank": 1, "Model": "DistilBERT", "Category": "Transformer", "Accuracy": 91.02},
    {"Rank": 2, "Model": "Logistic Regression", "Category": "Classical ML", "Accuracy": 89.95},
    {"Rank": 3, "Model": "Linear SVM", "Category": "Classical ML", "Accuracy": 89.11},
    {"Rank": 4, "Model": "Naive Bayes", "Category": "Classical ML", "Accuracy": 86.83},
    {"Rank": 5, "Model": "BiLSTM", "Category": "Deep Learning", "Accuracy": 77.07},
    {"Rank": 6, "Model": "LSTM", "Category": "Deep Learning", "Accuracy": 75.63},
]

st.set_page_config(
    page_title="CineMind | Movie Sentiment",
    page_icon="🎬",
    layout="wide",
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b0e14;
    --border: #263244;
    --text: #edf2fb;
    --muted: #96a3b8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 8% 0%, rgba(70,110,190,.13), transparent 34%),
        radial-gradient(ellipse at 95% 10%, rgba(137,92,205,.10), transparent 30%),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1220px;
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

[data-testid="stHeader"] {
    background: rgba(11,14,20,.7);
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    letter-spacing: -.035em;
    color: var(--text);
}

p, label, li {
    color: var(--muted);
}

.topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: .25rem 0 1.1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.2rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: .75rem;
    color: var(--text);
    font-family: 'Manrope', sans-serif;
    font-size: 1.03rem;
    font-weight: 800;
}

.brand-icon {
    display: grid;
    place-items: center;
    width: 2.45rem;
    height: 2.45rem;
    border-radius: .8rem;
    background: linear-gradient(135deg, #253b63, #392c5d);
    border: 1px solid #46577a;
    font-size: 1.25rem;
}

.status-pill {
    color: #9ee7c1;
    background: rgba(45,145,100,.12);
    border: 1px solid rgba(76,190,137,.3);
    padding: .4rem .7rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 700;
}

.eyebrow {
    color: #9dbbff;
    font-size: .75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .16em;
    margin-bottom: .7rem;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    color: var(--text);
    font-size: clamp(2.1rem, 5vw, 3.35rem);
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -.06em;
    margin: 0 0 .8rem;
}

.hero-title span {
    background: linear-gradient(90deg, #8bb7ff, #c0a5ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-copy {
    max-width: 720px;
    color: var(--muted);
    font-size: 1.02rem;
    line-height: 1.7;
    margin-bottom: 1.7rem;
}

.section-heading {
    color: var(--text);
    font-family: 'Manrope', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -.035em;
    margin: .2rem 0 .3rem;
}

.section-copy {
    color: var(--muted);
    font-size: .9rem;
    margin-bottom: 1rem;
}

.metric-card {
    min-height: 126px;
    padding: 1.1rem 1.15rem;
    background: linear-gradient(145deg, rgba(22,31,46,.96), rgba(15,20,30,.96));
    border: 1px solid var(--border);
    border-radius: 1rem;
}

.metric-label {
    color: var(--muted);
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
    margin-bottom: .75rem;
}

.metric-value {
    color: var(--text);
    font-family: 'Manrope', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -.045em;
}

.metric-note {
    color: #9baac0;
    font-size: .78rem;
    margin-top: .5rem;
}

div[data-testid="stTextArea"] textarea {
    background: #0e141e !important;
    border: 1px solid #2b394e !important;
    border-radius: .85rem !important;
    color: var(--text) !important;
    line-height: 1.65 !important;
    padding: 1rem !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #779ef0 !important;
    box-shadow: 0 0 0 1px #779ef0 !important;
}

.stButton > button {
    border-radius: .75rem;
    min-height: 3rem;
    font-weight: 800;
    border: 1px solid #8aabf8;
    background: linear-gradient(100deg, #739cf0, #9b87ed);
    color: #0b1020;
}

.stButton > button:hover {
    filter: brightness(1.08);
    border-color: #b7caff;
    color: #080c16;
}

.result-card {
    border-radius: 1rem;
    border: 1px solid var(--border);
    background: linear-gradient(145deg, rgba(19,28,42,.98), rgba(13,18,27,.98));
    padding: 1.35rem;
    margin-top: .8rem;
}

.result-positive {
    border-left: 4px solid #50d6a0;
}

.result-negative {
    border-left: 4px solid #fb7185;
}

.result-label {
    color: var(--muted);
    text-transform: uppercase;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: .12em;
}

.result-value {
    color: var(--text);
    font-family: 'Manrope', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: .35rem 0;
}

.result-detail {
    color: var(--muted);
    font-size: .88rem;
    line-height: 1.6;
}

.confidence-track {
    height: 7px;
    background: #283345;
    border-radius: 999px;
    overflow: hidden;
    margin: .8rem 0 .4rem;
}

.confidence-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #739cf0, #a78bfa);
}

.table-wrap {
    border: 1px solid var(--border);
    border-radius: .9rem;
    overflow: hidden;
}

.table-head, .table-row {
    display: grid;
    grid-template-columns: .55fr 2.2fr 1.5fr 1fr;
    gap: .7rem;
    align-items: center;
    padding: .9rem 1rem;
}

.table-head {
    background: #171f2d;
    color: #9eacc1;
    font-size: .72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.table-row {
    border-top: 1px solid #222d3d;
    color: #e9eef8;
    font-size: .9rem;
}

.table-row:nth-child(2) {
    background: rgba(104,143,231,.08);
}

.rank {
    color: #9bb9ff;
    font-weight: 800;
}

.accuracy {
    font-weight: 800;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

.category-tag {
    color: #b7c5da;
    font-size: .76rem;
}

.best-tag {
    display: inline-block;
    color: #a8f0cf;
    background: rgba(51,171,119,.13);
    border: 1px solid rgba(81,194,143,.25);
    border-radius: 999px;
    padding: .2rem .5rem;
    margin-left: .45rem;
    font-size: .66rem;
    font-weight: 800;
}

.footer {
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    color: #7f8da2;
    font-size: .78rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: .6rem;
}

@media (max-width: 700px) {
    .table-head, .table-row {
        grid-template-columns: .45fr 1.7fr 1fr;
    }

    .table-head div:nth-child(3),
    .table-row div:nth-child(3) {
        display: none;
    }

    .metric-value {
        font-size: 1.45rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    return tokenizer, model


# ============================================================
# PREDICTION
# ============================================================

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0, predicted_class].item())

    return {0: "Negative", 1: "Positive"}[predicted_class], confidence


def explain_prediction(text, tokenizer, model, n_steps=20):
    """
    Estimate token-level contributions to the model's predicted class
    using Layer Integrated Gradients.
    """

    model.eval()
    device = next(model.parameters()).device

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    def forward_func(ids, mask):
        return model(
            input_ids=ids,
            attention_mask=mask,
        ).logits

    with torch.no_grad():
        logits = forward_func(input_ids, attention_mask)
        predicted_class = int(logits.argmax(dim=1).item())

    # Use padding-token IDs as the baseline, while retaining special tokens.
    baseline_ids = torch.full_like(
        input_ids,
        tokenizer.pad_token_id,
    )

    special_ids = {
        token_id
        for token_id in tokenizer.all_special_ids
        if token_id is not None
    }

    for index, token_id in enumerate(input_ids[0].tolist()):
        if token_id in special_ids:
            baseline_ids[0, index] = token_id

    layer = model.get_input_embeddings()

    lig = LayerIntegratedGradients(
        forward_func,
        layer,
    )

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=predicted_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Sum attribution across the embedding dimension.
    token_scores = attributions.sum(dim=-1)[0].detach().cpu()
    token_ids = input_ids[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    rows = []

    for token, score, token_id in zip(
        tokens,
        token_scores.tolist(),
        token_ids,
    ):
        if token_id in special_ids:
            continue

        rows.append({
            "Token": tokenizer.convert_tokens_to_string([token]),
            "Contribution": score,
            "Interpretation": (
                "Supports predicted class"
                if score > 0
                else "Opposes predicted class"
                if score < 0
                else "Near neutral"
            ),
        })

    # Largest absolute contributions first.
    rows.sort(
        key=lambda row: abs(row["Contribution"]),
        reverse=True,
    )

    return predicted_class, rows, float(delta.abs().max().item())



# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="topline">
  <div class="brand">
    <div class="brand-icon">🎬</div>
    <div>CineMind <span style="color:#8493aa;font-weight:500;">/ Research Lab</span></div>
  </div>
  <div class="status-pill">● MODEL READY</div>
</div>

<div class="eyebrow">NLP · MACHINE LEARNING · TRANSFORMERS</div>
<div class="hero-title">Understand the feeling<br>behind every <span>review.</span></div>
<div class="hero-copy">
An intelligent movie review analyzer powered by a fine-tuned DistilBERT transformer.
Enter a review to classify its sentiment and inspect the model's confidence.
</div>
""", unsafe_allow_html=True)


# ============================================================
# PERFORMANCE CARDS
# ============================================================

metric_cols = st.columns(4)

metrics = [
    ("BEST MODEL ACCURACY", "91.02%", "DistilBERT · held-out test set"),
    ("REVIEWS EVALUATED", "25,000", "IMDb test reviews"),
    ("MODELS COMPARED", "06", "Classical ML, RNNs & Transformer"),
    ("TASK", "Binary", "Positive or negative sentiment"),
]

for col, (label, value, note) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# ============================================================
# SENTIMENT ANALYZER
# ============================================================

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown(
        '<div class="section-heading">Analyze a movie review</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-copy">Paste a review below. Inputs are truncated to 256 tokens.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        review = st.text_area(
            "Your review",
            height=220,
            placeholder=(
                "The performances were outstanding, the story kept me engaged, "
                "and the ending was genuinely moving..."
            ),
            label_visibility="collapsed",
            key="review_input",
        )

        word_count = len(review.split()) if review.strip() else 0
        char_count = len(review) if review else 0

        st.caption(f"{word_count:,} words  ·  {char_count:,} characters")

        analyze = st.button(
            "Analyze sentiment  →",
            type="primary",
            use_container_width=True,
        )

    if analyze:
        if not review.strip():
            st.warning("Enter a movie review before running the analysis.")
        else:
            try:
                with st.spinner("Reading the review..."):
                    tokenizer, model = load_model()
                    sentiment, confidence = predict_sentiment(
                        review, tokenizer, model
                    )

                st.session_state["last_prediction"] = {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "review": review,
                }

            except Exception:
                st.error("The model could not be loaded or prediction failed.")
                st.exception(Exception("Check the traceback above."))


with right:
    st.markdown(
        '<div class="section-heading">Prediction result</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-copy">Your latest analysis appears here.</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.get("last_prediction")

    if result:
        positive = result["sentiment"] == "Positive"
        sentiment = result["sentiment"]
        confidence_pct = result["confidence"] * 100

        icon = "✦" if positive else "−"
        css_class = "result-positive" if positive else "result-negative"

        explanation = (
            "The model detected language patterns associated with a positive review."
            if positive
            else "The model detected language patterns associated with a negative review."
        )

        st.markdown(
            f"""
            <div class="result-card {css_class}">
              <div class="result-label">Predicted sentiment</div>
              <div class="result-value">{icon} {sentiment}</div>
              <div class="result-detail">{explanation}</div>

              <div style="margin-top:1.4rem;display:flex;justify-content:space-between;gap:1rem;">
                <span class="result-label">Model confidence</span>
                <strong style="color:#edf2fb;">{confidence_pct:.2f}%</strong>
              </div>

              <div class="confidence-track">
                <div class="confidence-fill" style="width:{confidence_pct:.2f}%;"></div>
              </div>

              <div class="result-detail">DistilBERT · maximum sequence length 256 tokens</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Review submitted"):
            st.write(result["review"])

        st.caption(
            "Confidence is the model's probability for its selected class, "
            "not a guarantee of correctness."
        )

    else:
        with st.container(border=True):
            st.markdown("#### Ready when you are")
            st.write("Submit a review to see its predicted sentiment and confidence here.")
            st.write("**Model:** Fine-tuned DistilBERT")
            st.write("**Output:** Positive or Negative")
            st.write("**Input limit:** 256 tokens after tokenization")


# ============================================================
# EXPLAIN PREDICTION
# ============================================================

explain_tab, = st.tabs(["Explain Prediction"])

with explain_tab:
    st.subheader("Why did the model make this prediction?")
    st.caption(
        "Integrated Gradients estimates how individual tokens influence "
        "the selected class score. These are model attributions, not proof "
        "of human-like reasoning."
    )

    result = st.session_state.get("last_prediction")

    if result is None:
        st.info(
            "Analyze a review first. Its explanation will appear here."
        )
    else:
        if st.button(
            "Generate explanation",
            key="generate_explanation",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Calculating token contributions..."
                ):
                    tokenizer, model = load_model()

                    predicted_class, rows, delta = explain_prediction(
                        result["review"],
                        tokenizer,
                        model,
                    )

                    st.session_state["last_explanation"] = {
                        "review": result["review"],
                        "predicted_class": predicted_class,
                        "rows": rows,
                        "delta": delta,
                    }

            except Exception as error:
                st.error("Could not generate the explanation.")
                st.exception(error)

        explanation = st.session_state.get("last_explanation")

        if explanation and explanation["review"] == result["review"]:
            sentiment_name = {
                0: "Negative",
                1: "Positive",
            }[explanation["predicted_class"]]

            st.markdown(f"**Predicted sentiment: {sentiment_name}**")
            st.write(
                "Tokens with positive scores support the selected class; "
                "negative scores oppose it."
            )

            if explanation["rows"]:
                explanation_df = pd.DataFrame(explanation["rows"])
                st.dataframe(
                    explanation_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Token": st.column_config.TextColumn("Token"),
                        "Contribution": st.column_config.NumberColumn(
                            "Attribution score",
                            format="%.5f",
                        ),
                        "Interpretation": st.column_config.TextColumn(
                            "Estimated influence"
                        ),
                    },
                )

            st.caption(
                f"Integrated Gradients convergence delta: "
                f"{explanation['delta']:.6f}. "
                "This is a numerical diagnostic, not an accuracy score."
            )

# ============================================================
# MODEL COMPARISON — NATIVE TABLE, NO CHARTS
# ============================================================

st.write("")

st.markdown(
    '<div class="section-heading">Model benchmark</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-copy">'
    'Accuracy reported from the 25,000-review IMDb test set.'
    '</div>',
    unsafe_allow_html=True,
)

comparison = pd.DataFrame([
    {"Rank": 1, "Model": "DistilBERT", "Category": "Transformer", "Accuracy": "91.02%"},
    {"Rank": 2, "Model": "Logistic Regression", "Category": "Classical ML", "Accuracy": "89.95%"},
    {"Rank": 3, "Model": "Linear SVM", "Category": "Classical ML", "Accuracy": "89.11%"},
    {"Rank": 4, "Model": "Naive Bayes", "Category": "Classical ML", "Accuracy": "86.83%"},
    {"Rank": 5, "Model": "BiLSTM", "Category": "Deep Learning", "Accuracy": "77.07%"},
    {"Rank": 6, "Model": "LSTM", "Category": "Deep Learning", "Accuracy": "75.63%"},
])

st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Rank": st.column_config.NumberColumn("Rank", format="%d"),
        "Model": st.column_config.TextColumn("Model"),
        "Category": st.column_config.TextColumn("Category"),
        "Accuracy": st.column_config.TextColumn("Accuracy"),
    },
)

with st.expander("About the benchmark"):
    st.write(
        "DistilBERT achieved the highest measured accuracy among the six "
        "evaluated models. Results are specific to this dataset and evaluation "
        "setup; accuracy on new reviews may differ."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
  <div>Advanced Movie Sentiment Analysis</div>
  <div>Built with Streamlit · PyTorch · Hugging Face Transformers</div>
</div>
""", unsafe_allow_html=True)
