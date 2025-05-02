from flask import Flask, render_template, request
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('stopwords')

# Nettoyage
stop_words_fr = set(stopwords.words('french'))

def nettoyer_texte(texte):
    if pd.isnull(texte):
        return ""
    texte = texte.lower()
    texte = texte.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(texte, language='french')
    tokens = [mot for mot in tokens if mot not in stop_words_fr]
    return ' '.join(tokens)

# Chargement des données
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Prétraitement
train["text_clean"] = train["text"].apply(nettoyer_texte)
X_train = train["text_clean"]
y_train = train["sentiment"]

vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vect, y_train)

# Création de l'app Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    resultat = None
    if request.method == "POST":
        texte = request.form["texte"]
        texte_clean = nettoyer_texte(texte)
        vect = vectorizer.transform([texte_clean])
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]
        sentiments = model.classes_
        details = [(sentiments[i], f"{proba[i]*100:.2f}%") for i in range(len(sentiments))]
        resultat = {"prediction": pred, "details": details}
    return render_template("index.html", resultat=resultat)

if __name__ == "__main__":
    app.run(debug=True)
