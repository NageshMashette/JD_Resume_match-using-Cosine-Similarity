import nltk
nltk.download("punkt")
nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # Remove stop words and stem the remaining words
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Join the stemmed words back into a string
    preprocessed_text = " ".join(stemmed_words)

    return preprocessed_text

def get_similarity_score(jd_file_path, resume_file_path):
    # Read the text from the job description file
    with open(jd_file_path, "r", encoding="iso-8859-1") as jd_file:
        jd_text = jd_file.read()

    # Read the text from the resume file
    with open(resume_file_path, "r", encoding="iso-8859-1") as resume_file:
        resume_text = resume_file.read()

    # Preprocess the text
    jd_preprocessed = preprocess_text(jd_text)
    resume_preprocessed = preprocess_text(resume_text)

    # Use TF-IDF to vectorize the preprocessed text
    vectorizer = TfidfVectorizer()
    jd_vector = vectorizer.fit_transform([jd_preprocessed])
    resume_vector = vectorizer.transform([resume_preprocessed])

    # Calculate the cosine similarity between the vectors for the two texts
    sim_score = cosine_similarity(jd_vector, resume_vector)[0][0]

    return sim_score




jd_file_path = "C:/Users/n72/Desktop/JD.txt"
resume_file_path = "C:/Users/n72/Desktop/46196036_EN_Nagesh Mashette_datascience.txt"

similarity_score = get_similarity_score(jd_file_path, resume_file_path)
print("Similarity score:", similarity_score)
