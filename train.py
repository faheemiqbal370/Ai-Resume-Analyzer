# train.py (sketch)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report


# load labeled dataset
# df has columns: resume_text, skill_python, skill_sql, skill_git, ...
df = pd.read_csv('labeled_resumes.csv')
X = df['resume_text'].values
Y = df.drop(columns=['resume_text']).values


vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
Xvec = vectorizer.fit_transform(X)


Xtr, Xte, ytr, yte = train_test_split(Xvec, Y, test_size=0.2, random_state=42)
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(Xtr, ytr)


pred = clf.predict(Xte)
print('micro F1:', f1_score(yte, pred, average='micro'))
print(classification_report(yte, pred))


# Save model using joblib
import joblib
joblib.dump(clf, 'skill_clf.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')