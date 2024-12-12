from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

app = Flask(__name__)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Sample questions and answers
faq = {
    "What courses are available?": "We offer a variety of undergraduate and postgraduate courses in Arts, Science, and Engineering.",
    "How do I apply for admission?": "You can apply for admission through our online portal on the university website.",
    "Where is the campus located?": "Our campus is located in the heart of the city, at 123 University Ave.",
    "What extracurricular activities are available?": "We have clubs for sports, music, arts, and technology, along with volunteering opportunities."
}

# Preprocess questions
questions = list(faq.keys())
answers = list(faq.values())

def preprocess(text):
    """Tokenize and clean text using spaCy."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Preprocess stored questions
preprocessed_questions = [preprocess(q) for q in questions]

# Vectorize questions
vectorizer = TfidfVectorizer()
vectorizer.fit(preprocessed_questions)
question_vectors = vectorizer.transform(preprocessed_questions)

@app.route("/", methods=["GET", "POST"])
def chatbot():
    response = ""
    if request.method == "POST":
        user_question = request.form.get("question")
        user_preprocessed = preprocess(user_question)
        user_vector = vectorizer.transform([user_preprocessed])

        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, question_vectors)
        max_similarity = max(similarities[0])

        # Match threshold
        threshold = 0.5
        if max_similarity > threshold:
            index = similarities[0].argmax()
            response = answers[index]
        else:
            response = "Sorry, I don't understand that question."
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
