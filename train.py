import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

data = pd.read_csv("final_data.csv")
test = pd.read_csv("test - Sheet1.csv")
texts = data["text"].astype(str)
test_texts = test["tweets"].astype(str)
y = data["is_offensive"]

vectorizer = TfidfVectorizer(stop_words="english", min_df=0.0001)
X = vectorizer.fit_transform(texts)

model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y)

def predict_prob(texts):
  return cclf.predict_proba(vectorizer.transform(texts))[:,1]
racial_slur_degree = []

racial_slur_degree = predict_prob(test_texts)


test['output'] = racial_slur_degree
test.to_csv (r'Output.csv', index = False, header=True)


