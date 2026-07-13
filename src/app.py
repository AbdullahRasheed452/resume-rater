import gradio as gr
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load our trained model and vectorizer (saved earlier)
classifier_model = joblib.load('classifier_model.pkl')
classifier_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def analyze_resume(resume_text, job_description):
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description)
    
    # Predict category
    resume_vector = classifier_vectorizer.transform([cleaned_resume])
    predicted_category = classifier_model.predict(resume_vector)[0]
    
    # Match score (fresh TF-IDF just for these two texts)
    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform([cleaned_jd, cleaned_resume])
    match_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    
    # Missing keywords
    feature_names = vec.get_feature_names_out()
    jd_scores = tfidf[0].toarray()[0]
    resume_scores = tfidf[1].toarray()[0]
    
    GENERIC_WORDS = {'ideal', 'looking', 'strong', 'candidate', 'background', 
                      'handling', 'specialist', 'experience', 'skills'}
    
    missing = []
    for i, word in enumerate(feature_names):
        if jd_scores[i] > 0 and resume_scores[i] == 0 and word not in GENERIC_WORDS:
            missing.append((word, jd_scores[i]))
    missing.sort(key=lambda x: x[1], reverse=True)
    missing_keywords = [word for word, score in missing[:10]]
    
    result = f"## Predicted Category: {predicted_category}\n\n"
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