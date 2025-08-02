from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the trained classification model
model = joblib.load('model.pkl')

# Load fact-verified dataset (True.csv must exist in the same folder)
true_df = pd.read_csv('True.csv')
fact_base = (true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')).str.strip()

# Vectorize verified facts
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
fact_vectors = vectorizer.fit_transform(fact_base)

# Function to find the closest fact match
def find_best_match(user_input):
    input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vec, fact_vectors).flatten()
    best_match_idx = similarities.argmax()
    score = similarities[best_match_idx]
    return fact_base.iloc[best_match_idx], round(score * 100, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('news', '').strip()
    if not text:
        return render_template('index.html', prediction="Please paste a news article.")

    # Predict fake or real
    pred = model.predict([text])[0]
    result = "✅ Likely Real" if pred == 1 else "❌ Likely Fake"

    # If real, find closest verified article
    matched_fact, similarity = find_best_match(text) if pred == 1 else (None, None)

    return render_template(
        'index.html',
        prediction=result,
        original=text,
        matched_fact=matched_fact,
        similarity=similarity
    )

if __name__ == '__main__':
    app.run(debug=True)
