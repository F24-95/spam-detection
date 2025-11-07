import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources (only runs once)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(y)

# Sample training data
texts = ["free money now", "hello friend", "win cash instantly", "good morning", "buy this product", "how are you"]
labels = [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham

# Preprocess and train model
processed_texts = [transform_text(t) for t in texts]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)

model = MultinomialNB()
model.fit(X, labels)

# Streamlit UI
st.title("SMS Spam Classifier")

msg = st.text_area("Enter your message:")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed = transform_text(msg)
        vector = vectorizer.transform([transformed])
        result = model.predict(vector)[0]

        if result == 1:
            st.error("Spam!")
        else:
            st.success("Not Spam.")