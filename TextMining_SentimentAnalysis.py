#!/usr/bin/env python
# coding: utf-8

# # **Task 3**

# In[1]:


get_ipython().system('pip3 install wordcloud')
get_ipython().system('pip install textblob')


# #### Loading Libraries

# In[2]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

import string
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re

import nltk
nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4',
               'vader_lexicon'])


# #### Importing dataset and Exploration

# In[3]:


df = pd.read_csv('sentiment.csv', encoding='ISO-8859-1')
df.head(10)


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe().T


# In[7]:


df.shape


# In[8]:


df['sentiment'].value_counts()


# #### Handling Missing value and Data Cleaning

# In[9]:


# checking for missing values. two coulmuns have missing values
df.isnull().sum()


# In[10]:


# filling missing value using the fillna function.
df.fillna('', inplace=True)
df.shape


# In[11]:


#checking after filling missing value 
df.isnull().sum()


# In[12]:


# checkinh for duplucated value.no duplicate found.
df.duplicated().sum()


# ### TEXT MINING AND PREPROCESSING

# #### Making statement text in lower case

# In[13]:


df['text']=df['text'].str.lower()
df['text'].head()


# #### Cleaning and removing  stop words list from the text

# In[14]:


STOPWORDS = set(stopwords.words('english'))
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
df['text'] = df['text'].apply(lambda text: cleaning_stopwords(text))
df['text'].head()


# #### Cleaning and removing punctuations

# In[15]:


import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
df['text']= df['text'].apply(lambda x: cleaning_punctuations(x))
df['text'].head()


# #### Cleaning and removing repeating characters

# In[16]:


def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
df['text'] = df['text'].apply(lambda x: cleaning_repeating_char(x))
df['text'].head(10)


# #### Cleaning and removing URL’s

# In[17]:


def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
df['text'] = df['text'].apply(lambda x: cleaning_URLs(x))
df['text'].head()


# In[18]:


# remove numerical number
def cleaning_numbers(data):
    return re.sub('[0-9]+', ' ', data)
df['text'] = df['text'].apply(lambda x: cleaning_numbers(x))
df['text'].head()


# #### Remove short words

# In[19]:


def transform_text(text):
    return ' '.join([word for word in text.split() if len(word) > 2])
df['text'] = df['text'].apply(lambda x: transform_text(x))
df['text'].head() 


# #### Tokenization

# In[20]:


# Function which directly tokenize the text data
from nltk.tokenize import TweetTokenizer

tt = TweetTokenizer()
df['text']=df['text'].apply(tt.tokenize)
df['text'].head()


# #### Stemming

# In[21]:


# stemming
import nltk
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
df['text']= df['text'].apply(lambda x: stemming_on_text(x))
df['text'].head()


# #### lemmatization

# In[22]:


lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
df['text'] = df['text'].apply(lambda x: lemmatizer_on_text(x))
df['text'].head()


# In[23]:


# create two new dataframe all of the positive text
df_positive = df[df['sentiment'] == 'positive']


# create two new dataframe all of the negative text
df_negative = df[df['sentiment'] == 'negative']


# create two new dataframe all of the neutral text
df_neutral=df[df['sentiment'] == 'neutral']


# #### EXploratory Data Analysis

# In[24]:


# graphical representation of sentiment value counts using the countplot from the seasonborn library
sns.countplot(data=df, x='sentiment')


# In[25]:


# show the value counts

df['sentiment'].value_counts()

#plot and visualize the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Count')
df['sentiment'].value_counts().plot(kind='bar')
plt.show()


# In[26]:


plt.figure(figsize = (8,4))
plt.title("Sentiment by User Age groups")
sns.countplot(x="Age of User", hue="sentiment", data = df)

plt.show()


# In[27]:


plt.figure(figsize = (8,4))
plt.title("Time of Tweet")
sns.countplot(x="sentiment", hue="Time of Tweet", data = df)

plt.show()


# #### WordCloud Visual for all Text

# In[28]:


all_words = " ".join(" ".join(sent) for sent in df['text'])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white',max_font_size=100).generate(all_words)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# #### WordCloud Visual for positive texts

# In[29]:


# Visualizing all positive tweets
               
all_pos_words = " ".join(" ".join(sent) for sent in df_positive['text'])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, background_color='white', max_font_size=100).generate(all_pos_words)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# #### WordCloud Visual for Negative texts

# In[30]:


# Visualizing all negative texts

all_neg_words = " ".join(" ".join(sent) for sent in df_negative['text'])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, background_color='white', max_font_size=100).generate(all_neg_words)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# #### WordCloud Visual for Neutral texts

# In[31]:


# Visualizing all neutral text

all_neu_words = " ".join(" ".join(sent) for sent in df_neutral['text'])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, background_color='white', max_font_size=100).generate(all_neu_words)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ####  Sentiments distribution in Percantage

# In[32]:


# Get the percentage of positive texts
print("Positive texts",round((df_positive.shape[0]/df.shape[0])*100,1),"%")

# Get the percentage of negative texts
print("Negative texts",round((df_negative.shape[0]/df.shape[0])*100,1),"%")

# Get the percentage of neutral texts
print("Neutral texts",round((df_neutral.shape[0]/df.shape[0])*100,1),"%")


# In[33]:


lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text
df['preprocessed_text'] = df['text'].astype(str).apply(preprocess_text)


# In[34]:


df['preprocessed_text'] = df['text'].astype(str).apply(preprocess_text)


# In[36]:


df[['sentiment','text', 'preprocessed_text']].head()


# #### VADER's Sentiment Intensity Analyzer

# In[37]:


sia = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    return sia.polarity_scores(text)['compound']

df['vader_sentiment'] = df['preprocessed_text'].apply(get_vader_sentiment)

df['vader_sentiment_category'] = df['vader_sentiment'].apply(
    lambda score: 'positive' if score > 0 else ('negative' if score < 0 else 'neutral'))


# In[38]:


print(df[['text', 'preprocessed_text', 'vader_sentiment', 'vader_sentiment_category']].head())


# In[39]:


sentiment_distribution = df['vader_sentiment'].value_counts(normalize=True) * 100

sentiment_distribution


# #### Text Preprocessing and Training a Naïve Bayes classifier

# In[40]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(df['text'].map(' '.join))

X.shape


# In[41]:


from sklearn.model_selection import train_test_split

y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size=0.2, random_state=0)
print('Training Data :', X_train.shape)
print('Testing Data : ', X_test.shape)


# In[42]:


# handling imbalance using the RandomUndersampler 
from imblearn.under_sampling import RandomUnderSampler

resampler = RandomUnderSampler(random_state=0)
X_train_undersampled,y_train_undersampled = resampler.fit_resample(X_train,y_train)

sns.countplot(x=y_train_undersampled)


# #### Training model using the Naive Bayes Classifier

# In[43]:


# training model using the naive bayes model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_undersampled, y_train_undersampled)


# In[44]:


y_pred = model.predict(X_test)
y_pred


# #### Computing the Accuracy and making the Confusion Matrix

# In[45]:


y_pred = model.predict(X_test)
# computing the accuracy and making the confusion matrix

from sklearn import metrics
acc = metrics.accuracy_score(y_test, y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm,'\n\n')
print('--------------------------------------------------')
result = metrics.classification_report(y_test, y_pred)
print("Classification Report:\n",)
print(result)


# In[46]:


ax = sns.heatmap(cm, cmap = 'flare', annot = True, fmt = 'd')

plt.title("confusion Matrix", fontsize = 12)
plt.xlabel("Predicted Class", fontsize = 12)
plt.ylabel("True Class", fontsize = 12)
plt.show()


# In[ ]:




