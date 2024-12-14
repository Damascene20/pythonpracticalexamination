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
    "Does the university provide career services?": "Yes, our career center offers resume workshops, interview preparation, and job placement assistance.",
    "What is the process to drop a class?": "You can drop a class through the student portal before the deadline specified in the academic calendar.",
    "What is the student-to-faculty ratio?": "Our student-to-faculty ratio is 15:1, allowing for a more personalized learning experience.",
    "What is the average class size?": "The average class size is around 30 students, but this can vary depending on the course.",
    "What are the admission requirements for international students?": "International students need to provide transcripts, proof of English proficiency, and an updated passport.",
    "Are there on-campus job opportunities?": "Yes, we offer on-campus work-study programs for eligible students.",
    "How do I get involved in research projects?": "You can join research projects through faculty members or the research office. Check the research opportunities page.",
    "What are the university’s policies on academic integrity?": "The university has strict policies on academic honesty, including zero tolerance for plagiarism and cheating.",
    "Is there a student health center?": "Yes, we have a health center on campus that provides medical care and wellness services.",
    "Can I park on campus?": "Yes, we offer parking permits for students, staff, and visitors. There are designated parking areas around the campus.",
    "How do I get a library card?": "All students are automatically registered for library services. You can access the library using your student ID.",
    "What is the student discount for local transportation?": "Students can get discounted transportation passes by showing their student ID at the transport office.",
    "Is there a shuttle service to the airport?": "Yes, we offer a shuttle service to the nearest airport at the start and end of each semester.",
    "What is the process for transferring credits from another university?": "You need to submit official transcripts for evaluation. The transfer credit office will determine which courses can be transferred.",
    "How do I access counseling services?": "Counseling services are available through the student wellness center. You can book an appointment through the online portal.",
    "Can I change my major?": "Yes, you can change your major by meeting with an academic advisor and submitting a change of major form.",
    "Are there any language support services for international students?": "We offer English language support programs for international students to help with writing, speaking, and comprehension.",
    "What student organizations are available?": "We have a wide variety of student organizations, including cultural clubs, academic societies, and volunteer groups.",
    "Is there a university bookstore?": "Yes, our campus bookstore sells textbooks, stationery, university merchandise, and other academic supplies.",
    "Can I request a letter of recommendation from a professor?": "Yes, you can request a letter of recommendation by contacting your professor well in advance.",
    "What are the campus security measures?": "We have 24/7 campus security, including surveillance cameras, emergency call boxes, and a campus police department.",
    "Are there any religious services on campus?": "Yes, we offer a variety of religious services and have a designated multi-faith space on campus.",
    "What is the university's policy on mental health?": "The university offers various mental health resources, including counseling, workshops, and stress-relief activities.",
    "What is the process for graduating?": "To graduate, you must complete all required coursework, meet credit requirements, and submit a graduation application before the deadline.",
    "How do I access my grades?": "You can view your grades through the student portal after they have been posted by your professors.",
    "What is the policy on late submissions?": "Late submissions are generally penalized unless there are extenuating circumstances. Check the course syllabus for specific policies.",
    "What are the options for studying part-time?": "We offer part-time study options for many programs. Contact the academic office for details.",
    "Can I take online courses?": "Yes, we offer several online courses and fully online programs in various fields.",
    "How do I request accommodations for a disability?": "You can request accommodations by contacting the student disability services office and providing the necessary documentation.",
    "Is there a student government?": "Yes, we have a student government association that represents the student body and organizes events.",
    "Can I get a letter for a visa application?": "Yes, you can request a visa letter from the admissions office to assist with your visa application process.",
    "How do I appeal my grades?": "You can appeal your grades by submitting a formal request through the academic office within a specified period.",
    "Is there a career fair?": "Yes, we host annual career fairs where students can meet employers and learn about job opportunities.",
    "How do I get involved in community service?": "You can get involved in community service through student organizations, campus events, and partnerships with local nonprofits.",
    "How do I apply for a graduate program?": "Graduate applications are submitted through our graduate admissions portal. Ensure you meet the specific program requirements.",
    "Are there any opportunities for entrepreneurship?": "Yes, we offer an entrepreneurship center that supports student startups with resources, mentorship, and funding.",
    "How do I pay my tuition fees?": "Tuition fees can be paid online through the student portal or in person at the university's financial office.",
    "Can I use the gym without a membership?": "Access to the gym is free for students with a valid ID. There are additional services available with a premium membership.",
    "What is the campus culture like?": "Our campus culture is diverse, inclusive, and community-focused. We encourage collaboration and open-mindedness.",
    "What types of financial aid are available?": "We offer a variety of financial aid options, including scholarships, grants, and student loans.",
    "How do I get involved in arts and culture events?": "You can participate in arts and culture events by joining student-run clubs, attending performances, and volunteering for events.",
    "Can I take a leave of absence?": "Yes, you can take a leave of absence for personal or medical reasons by submitting a request to the academic office.",
    "What is the average cost of living on campus?": "The average cost of living on campus varies by accommodation type, but it generally ranges from $500 to $1,200 per month.",
    "What are the university's sustainability initiatives?": "We have several sustainability initiatives, including waste reduction programs, energy conservation, and a campus garden.",
    "Is there an alumni network?": "Yes, our alumni network offers events, mentorship opportunities, and job connections for graduates.",
    "How can I get involved in research as an undergraduate?": "Undergraduates can get involved in research by contacting faculty members or applying for undergraduate research programs.",
    "What is the process for applying for housing?": "You can apply for on-campus housing through the housing application portal, typically during the summer before the academic year begins.",
    "How can I access online learning materials?": "Online learning materials are available through the student portal. Check your course page for specific materials.",
    "What is the university's policy on social media?": "We encourage students to use social media responsibly and respectfully, in accordance with the university's code of conduct.",
    "How do I find academic advising?": "Academic advising is available through the academic office. You can schedule an appointment through the student portal.",
    "What are the university's policies on class attendance?": "Class attendance policies vary by course and professor. Generally, regular attendance is expected for all classes.",
    "How do I request an official transcript?": "Official transcripts can be requested through the student portal or by contacting the registrar's office.",
    "What is the process for changing my contact information?": "You can update your contact information through the student portal or by contacting the registrar's office.",
    "Are there any opportunities to work with faculty on projects?": "Yes, students can work with faculty on various academic and research projects. Contact your professors for opportunities.",
    "Can I participate in study groups?": "Yes, many students organize study groups. You can also join study groups through student clubs or the tutoring center.",
    "What is the university's policy on alcohol and drugs?": "The university has a zero-tolerance policy for alcohol and drug use on campus. Violations are subject to disciplinary action."
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
