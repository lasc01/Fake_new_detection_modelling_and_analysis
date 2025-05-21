# 📰 Fake News Detection Through Textual Feature Analysis

This repository showcases a comprehensive machine learning project aimed at detecting fake news using textual features. It is part of a group-based academic assessment under the MSc in Data Analytics program at the National College of Ireland.

## 📌 Overview

- **Objective**: Develop a robust model to classify news articles as fake or real based on text features.
- **Contributors**: Liton, Olamide, Osama (Team Project)
- **Scope**: Data extraction, cleaning, analysis, feature engineering, machine learning, and visualization.

## 📊 Key Features

- ETL pipeline for structured and semi-structured data (CSV, API)
- NLP-based preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF vectorization and feature engineering (readability, sentiment, etc.)
- Machine learning classification with Random Forest
- Evaluation using Accuracy, Precision, Recall, F1-score
- Visualizations: word clouds, sentiment plots, readability distributions

## 🗂️ Project Structure

```
Fake-News-Detection/
├── data/                  # Raw and cleaned datasets
│   ├── raw/               # Original data
│   └── processed/         # Cleaned and transformed data
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── src/                   # Scripts for ETL, preprocessing, modeling
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # MIT License
└── .gitignore             # Files/folders to ignore
```

## 🧪 Results Summary

- Model Accuracy: **99.84%**
- Fake news often exhibits higher emotional tone and subjectivity
- Real news uses factual, neutral language and is more readable

## 🚀 Getting Started

### Prerequisites

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Run Preprocessing

```bash
python src/preprocessing/text_cleaner.py
```

### Train Model

```bash
python src/modeling/rf_model.py
```

## 🗃️ Data Sources

- [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- TruthSeeker 2023 Dataset
- Guardian News API
- Politifact Extracted Dataset

## 👥 Contributors

- Liton
- Olamide
- Osama

## 📚 References

[1] R. Ahmed et al., "Detecting Fake News Using NLP and Machine Learning", Procedia Computer Science, 2020.  
[2] X. Zhou and R. Zafarani, "A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities", ACM, 2020.  
[3] M. Shu et al., "Fake News Detection on Social Media: A Data Mining Perspective", ACM SIGKDD, 2017.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
