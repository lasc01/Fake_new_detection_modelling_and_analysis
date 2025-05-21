
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    df = pd.read_csv('data/processed/cleaned_truth_seeker_dataset.csv')
    X = df['cleaned_text']
    y = df['label']

    tfidf = TfidfVectorizer(max_features=5000)
    X_vec = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("Model Evaluation:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
