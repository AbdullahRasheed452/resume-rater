import gradio as gr
import joblib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load our trained model and vectorizer (saved earlier)
classifier_model = joblib.load('classifier_model.pkl')
classifier_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Calibrated based on test set analysis: predictions below this score
# are less reliable than a coin flip across categories, so we flag them
# as "uncertain" instead of forcing a guess.
CONFIDENCE_THRESHOLD = -0.3

GENERIC_WORDS = {'ideal', 'looking', 'strong', 'candidate', 'background',
                  'handling', 'specialist', 'experience', 'skills'}


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def predict_category(cleaned_resume):
    resume_vector = classifier_vectorizer.transform([cleaned_resume])
    decision_scores = classifier_model.decision_function(resume_vector)[0]
    confidence = np.max(decision_scores)
    predicted = classifier_model.classes_[np.argmax(decision_scores)]

    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain — doesn't clearly match a known category", confidence
    return predicted, confidence


def get_missing_keywords(jd_text, resume_text, top_n=10):
    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform([jd_text, resume_text])

    feature_names = vec.get_feature_names_out()
    jd_scores = tfidf[0].toarray()[0]
    resume_scores = tfidf[1].toarray()[0]

    missing = []
    for i, word in enumerate(feature_names):
        if jd_scores[i] > 0 and resume_scores[i] == 0 and word not in GENERIC_WORDS:
            missing.append((word, jd_scores[i]))
    missing.sort(key=lambda x: x[1], reverse=True)
    return [word for word, score in missing[:top_n]]


def analyze_resume(resume_text, job_description):
    if not resume_text.strip() or not job_description.strip():
        return "Please paste both a resume and a job description."

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description)

    # Predicted category, with confidence check
    category_display, confidence = predict_category(cleaned_resume)

    # Match score (fresh TF-IDF just for these two texts)
    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform([cleaned_jd, cleaned_resume])
    match_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    # Missing keywords
    missing_keywords = get_missing_keywords(cleaned_jd, cleaned_resume)

    result = f"## Predicted Category: {category_display}\n"
    result += f"*(confidence score: {confidence:.2f})*\n\n"
    result += f"## Match Score: {match_score:.1%}\n\n"
    result += f"## Missing Keywords:\n"
    result += ", ".join(missing_keywords) if missing_keywords else "None - great match!"

    return result


demo = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.Textbox(lines=10, label="Paste Resume Text"),
        gr.Textbox(lines=6, label="Paste Job Description")
    ],
    outputs=gr.Markdown(label="Analysis"),
    title="Resume Rater & Classifier",
    description="Paste a resume and job description to get a match score, missing keywords, and predicted job category."
)

if __name__ == "__main__":
    demo.launch()