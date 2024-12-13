from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import nltk
import random
nltk.download('wordnet')

app = Flask(__name__)

# Predefined questions and answers
faq = {
    "What courses are available?": "We offer undergraduate and postgraduate courses in Arts, Science, Engineering, Business, and Technology.",
    "How do I apply for admission?": "You can apply for admission through our online portal available on the university website.",
    "What are the admission requirements?": "Admission requirements vary by program. Generally, they include transcripts, recommendation letters, and a personal statement.",
    "What is the application deadline?": "Application deadlines vary by program and semester. Please check the university website for specific dates.",
    "Are scholarships available?": "Yes, we offer merit-based and need-based scholarships. Visit the financial aid section on our website for details.",
    "What is the tuition fee?": "Tuition fees vary by program and level of study. Please refer to the fee structure on the university website.",
    "How can I contact the admissions office?": "You can contact the admissions office via email at admissions@university.edu or call us at 555-123-4567.",
    "Where is the campus located?": "Our campus is located at 123 University Avenue, in the heart of the city.",
    "What are the housing options?": "We offer on-campus dormitories and assistance with off-campus housing.",
    "What extracurricular activities are available?": "Students can join clubs for sports, arts, music, technology, and volunteering.",
    "Does the university provide internships?": "Yes, we have partnerships with leading organizations to provide internship opportunities for students.",
    "What support services are available for students?": "We offer counseling, career services, academic advising, and health services.",
    "Is there a campus tour available?": "Yes, you can schedule a campus tour through the Visit Us section on our website.",
    "What dining options are available?": "We have multiple dining halls, cafes, and restaurants offering diverse cuisines on campus.",
    "What are the library hours?": "The library is open from 8:00 AM to 10:00 PM on weekdays and 10:00 AM to 6:00 PM on weekends.",
    "How do I access my student email?": "You can access your student email by logging in with your university credentials at email.university.edu.",
    "What is the grading system?": "We use a 4.0 GPA grading system. Specific grading policies are outlined in the academic handbook.",
    "Are there study abroad opportunities?": "Yes, we offer study abroad programs in over 30 countries. Visit the International Programs section for details.",
    "What transportation options are available?": "We provide shuttle services, parking permits, and support for public transportation.",
    "How do I register for classes?": "Class registration is done through the student portal. Check the academic calendar for registration dates.",
    "What sports facilities are available?": "We have a gym, swimming pool, basketball courts, and a soccer field, among other facilities.",
    "Does the university provide career services?": "Yes, our career center offers resume workshops, interview preparation, and job placement assistance."
}

# Preprocess questions
questions = list(faq.keys())
answers = list(faq.values())

def preprocess(text):
    """Tokenize and clean text using TextBlob."""
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Split the text into words
    lemmatized_words = [Word(word).lemmatize() for word in words if word.isalpha()]  # Lemmatize words
    return " ".join(lemmatized_words)

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

        # Calculate similarity
        similarities = (user_vector * question_vectors.T).toarray()[0]
        max_similarity = max(similarities)

        # Match threshold
        threshold = 0.5  # Adjust as needed
        if max_similarity > threshold:
            index = similarities.argmax()
            response = answers[index]
        else:
            random_question = random.choice(questions)
            response = f"Sorry, I don't understand that question. Did you mean: '{random_question}'?"

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)