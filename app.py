"""Streamlit interface for the phishing email classifier."""

from __future__ import annotations

import streamlit as st

from phishing_detector import PHISHING_LABEL, predict_email, train_model


st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_model():
    return train_model()


st.title("🛡️ Phishing Email Detector")
st.caption("A Random Forest classifier trained on TF-IDF features from the capstone dataset")

with st.sidebar:
    st.header("Model details")
    st.markdown(
        """
        **Model:** Random Forest  
        **Features:** TF-IDF (1,000 terms)  
        **Notebook test accuracy:** 95.25%  
        **Phishing recall:** 96%  

        The tuned Random Forest was selected because it had the best
        cross-validation accuracy among the evaluated email models.
        """
    )
    st.warning(
        "This is a portfolio demonstration, not a replacement for an email "
        "security product. Never open links or attachments solely because an "
        "email is classified as safe."
    )

st.write(
    "Paste the subject and body of an email below. The model looks for language "
    "patterns associated with phishing; it does not open links or attachments."
)

with st.form("email_form"):
    email_text = st.text_area(
        "Email content",
        height=260,
        placeholder=(
            "Subject: Urgent account verification required\n\n"
            "Paste the full email body here..."
        ),
    )
    submitted = st.form_submit_button("Analyze email", type="primary", use_container_width=True)

if submitted:
    if not email_text.strip():
        st.error("Paste some email text before running the analysis.")
    else:
        try:
            with st.spinner("Loading the model and analyzing the email..."):
                model_bundle = get_model()
                result = predict_email(model_bundle.pipeline, email_text)

            if result.label == PHISHING_LABEL:
                st.error("⚠️ Likely phishing email")
                st.write(
                    "Treat this message as suspicious. Verify the sender through a "
                    "trusted channel and avoid its links and attachments."
                )
            else:
                st.success("✅ Likely safe email")
                st.write(
                    "The language resembles safe messages in the training data, but "
                    "you should still verify unexpected requests."
                )

            st.metric("Estimated phishing probability", f"{result.phishing_probability:.1%}")
            st.progress(result.phishing_probability)
            st.caption(
                f"Model trained on {model_bundle.training_rows:,} unique labeled emails "
                "from this repository. Probabilities are model confidence estimates, "
                "not guarantees."
            )
        except Exception:
            st.error("The model could not complete the analysis. Please try again.")

with st.expander("How the prediction works"):
    st.write(
        "The app converts email text into TF-IDF features, then sends those features "
        "to the tuned Random Forest from the project's modeling analysis. Training "
        "and inference use one pipeline, so the same preprocessing is applied in both places."
    )
