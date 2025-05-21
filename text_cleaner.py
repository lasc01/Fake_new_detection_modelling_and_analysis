
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def main():
    df = pd.read_csv('data/processed/cleaned_truth_seeker_dataset.csv')
    df['cleaned_text'] = df['text'].apply(clean_text)
    df.to_csv('data/processed/cleaned_truth_seeker_dataset.csv', index=False)
    print("Preprocessing complete. Cleaned file saved.")

if __name__ == "__main__":
    main()
