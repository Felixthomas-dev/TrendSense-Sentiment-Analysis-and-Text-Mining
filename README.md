## Uncovering Twitter Trends with Sentiment Analysis and Text Mining 📊

## Table of Contents
- [1. Introduction](#1-introduction) 📝
- [2. Objectives](#2-objectives) 🎯
- [3. Datasets](#3-datasets) 📈
- [4. Methodology](#4-methodology) 🔍
  - [4.1 Data Cleaning](#41-data-cleaning) 🧹
  - [4.2 Text Mining Preprocessing](#42-text-mining-preprocessing) 🛠
  - [4.3 Data Visualization](#43-data-visualization) 📊
- [5. Implementation in Python](#5-implementation-in-python) 🐍
  - [5.1 Sentiment Analysis Method (VADER)](#51-sentiment-analysis-method-vader) 💬
  - [5.2 Visualization of Results](#52-visualization-of-results) 🎨
- [6. Naive Bayes Classifier Algorithm](#6-naive-bayes-classifier-algorithm) 🤖
  - [6.1 Training Model](#61-training-model) 🏋️‍♂️
  - [6.2 Model Result Visualization](#62-model-result-visualization) 📉
- [7. Conclusion](#7-conclusion) 📚
- [8. References](#8-references) 🔗
- [9. License](#9-license) ⚖️

## 1. Introduction 📝

The explosion of internet usage and the popularity of social media platforms like Twitter have empowered individuals to freely express their thoughts, opinions, and feelings. This project aims to leverage sentiment analysis and text mining techniques to uncover trends on Twitter.

## 2. Objectives 🎯

- Utilize NLTK for preprocessing tasks: tokenization, stopword removal, URL removal, stemming, and lemmatization.
- Perform sentiment analysis with NLTK's VADER SentimentIntensityAnalyzer.
- Visualize word frequencies using the wordcloud package.
- Train a Naive Bayes classifier for sentiment categorization.

## 3. Datasets 📈

The dataset includes text posts from social media, enriched with user metadata such as posting time, demographic info, and location. It features text content and sentiment ratings, among other analytical variables.

## 4. Methodology 🔍

### 4.1 Data Cleaning 🧹

- Utilize `fillna()` for missing texts.
- Check for duplications with `Duplicated().Sum()`.

### 4.2 Text Mining Preprocessing 🛠

Employing NLTK for:
- **Tokenization**: Breaking sentences into words.
- **Stemming**: Reducing words to their base form.
- **Lemmatization**: Consolidating inflected words.
- **Stopword Removal**: Eliminating trivial words.

### 4.3 Data Visualization 📊

Graphical representations of the dataset features in relation to text sentiment, including various sentiment distribution plots.

## 5. Implementation in Python 🐍

### 5.1 Sentiment Analysis Method (VADER) 💬

VADER, effective for social media text, categorizes sentiments into positive, negative, or neutral categories.

### 5.2 Visualization of Results 🎨

- **Word Clouds**: Highlighting frequent words in social media texts.
- **Sentiment Distribution**: Illustrating the proportions of each sentiment category.

## 6. Naive Bayes Classifier Algorithm 🤖

### 6.1 Training Model 🏋️‍♂️

The MultinomialNB classifier is employed post-feature selection and class balancing.

### 6.2 Model Result Visualization 📉

The model's accuracy, confusion matrix, and classification report are displayed.

## 7. Conclusion 📚

This project offers insights into Twitter sentiment trends, utilizing sentiment analysis and Naive Bayes classification, demonstrating a balanced sentiment spectrum.

## 8. Data source 🔗

Sentiment Analysis Dataset. Kaggle. [Access it here](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data?select=train.csv). 

## 9. License ⚖️

This project is open source and available under the [MIT License](LICENSE), promoting free use, modification, and distribution of the software, ensuring that contributions are welcomed and recognized.

