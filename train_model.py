import pandas as pd
import string
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('stopwords')

stop_words_fr = set(stopwords.words('french'))

def nettoyer_texte(texte):
    if pd.isnull(texte):
        return ""
    texte = texte.lower()
    texte = texte.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(texte, language='french')
    tokens = [mot for mot in tokens if mot not in stop_words_fr]
    return ' '.join(tokens)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

train['text_clean'] = train['text'].apply(nettoyer_texte)

X_train = train['text_clean']
y_train = train['sentiment']

vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=200)
model.fit(X_train_vect, y_train)

# Sauvegarde du modèle et du vectoriseur
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Modèle et vectoriseur sauvegardés.")
