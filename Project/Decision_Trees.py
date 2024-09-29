# -*- coding: utf-8 -*-
"""ML Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1We-1obQPjpcpxGhQC0E2W8yjvNfRs292
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# database.describe()
#  sns.heatmap(database.corr(), cmap = "YlGnBu"). || correlation between the classes ||

# if you want to do it using polars:

# import polars as pl
# splits = {'train': 'train.tsv', 'validation': 'validation.tsv', 'test': 'test.tsv'}
# df = pl.read_csv('hf://datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English/' + splits['train'], separator='\t')

splits = {'train': 'train.tsv', 'validation': 'validation.tsv', 'test': 'test.tsv'}

# Load the training dataset
df_train = pd.read_csv("hf://datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English/" + splits["train"], sep="\t")

# Load the test dataset
df_test = pd.read_csv("hf://datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English/" + splits["test"], sep="\t")

#label'd as fake new 0 ; real news 1

# **Dropping unwanted columns**


df_train.columns

df_train = df_train.drop(['Unnamed: 0', 'title', 'subject', 'date'], axis=1)

df_test = df_test.drop(['Unnamed: 0', 'title', 'subject', 'date'], axis=1)

df_train.columns

df_test.columns

#count of missing values
df_train.isnull().sum()

"""# Data is already suffled so we do not require to do so"""

df_train.head()

"""# **Preprocessing Text**
# Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr and links.

"""

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

df_train['text'] = df_train['text'].apply(wordopt)

df_test['text'] = df_test['text'].apply(wordopt)

df_train.head()

# Tokenization

x_train = df_train['text']  # Features for training
y_train = df_train['label']  # Labels for training
x_test = df_test['text']     # Features for testing
y_test = df_test['label']     # Labels for testing

from sklearn.feature_extraction.text import TfidfVectorizer

# Creating an instance of TfidfVectorizer
vectorization = TfidfVectorizer()

# Fitting the vectorizer on the training data and transforming it into TF-IDF vectors
xv_train = vectorization.fit_transform(x_train)

# Transforming the test data into TF-IDF vectors
xv_test = vectorization.transform(x_test)

# Decision trees

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Initialize the Decision Tree Classifier
DT = DecisionTreeClassifier()

# Fit the model on the training data
DT.fit(xv_train, y_train)

# Make predictions on the test set
pred_dt = DT.predict(xv_test)

# Accuracy score
accuracy = DT.score(xv_test, y_test)
print(f'Accuracy: {accuracy}')

# Classification report for the predictions
print(classification_report(y_test, pred_dt))

# Manual Testing
# I entered news from the Validation.tsv file.

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"

def manual_testing(news):
    # Creating a DataFrame for the input news
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)

    # Apply Preprocessing function
    new_def_test['text'] = new_def_test["text"].apply(wordopt)

    # Prepare the text for prediction
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    # Predictions
    pred_DT = DT.predict(new_xv_test)

    # Print the results
    print("\nDT Prediction: {}".format(output_label(pred_DT[0])))

# this is a fake news
news_article = "Mexico has been the beneficiary of our open borders for decades. It s really quite amazing how loudly they cry foul when we finally have a presidential candidate willing to stand up to this insanity and say, ENOUGH! It s Time to put Americans first! Illegal immigrants send home $50 billion annually but cost taxpayers more than $113 billion. Approximately 126,000 illegal immigrants emigrated from these three nations to the U.S. since last October and federal officials estimate at least 95,500 more will enter next year.The Central American governments have encouraged the high levels of emigration because it is earning their economy billions of dollars! For every illegal alien that sneaks into the U.S. and remits money back home, that grand total remittance number only grows. But what if the millions of U.S. jobs now filled by illegal aliens were done by American workers earning better wages, paying more in taxes and spending their money in their communities rather than sending it abroad?Americans are the ones forced to pick up the $113 billion tab for taking care of the country s 12 million illegal immigrants. Is it the responsibility of taxpaying citizens to cover the cost of illegal immigration and the government s aid to these countries while illegal workers continue to send their money overseas to send $50 billion overseas? Via: immigrationreform.comYet, somehow Donald Trump is to BLAME for wanting to shut down our open borders and bring some sanity back to our nation that is being overrun by Mexican drug runners and gang bangers coming from every corner of Mexico, Central and South America? A new Mexican movie promoted by Univision host Jorge Ramos portrays a drunk vigilante motivated by Republican presidential candidate Donald Trump s anti-immigrant rant killing at least four illegal immigrants at the border.The trailer for the movie, Desierto, now in Mexican theaters, blasts out Trump s initial criticism of illegal immigrants as a man armed with a rifle guns down targets crossing under barbed wire.https://youtu.be/V48ttgGqsswIt later shows the same man, in a pickup holding a bottle of what appears to be whisky, a beer can nearby, as a voice says, Welcome to the land of the free. The trailer ends with, words are as dangerous as bullets. The Center for Immigration Studies on Tuesday first blogged on the exploitation movie and the promotion by Ramos on his Sunday show Al Punto. This material resonates powerfully with Jorge Ramos. His conviction that racism and xenophobia are the driving forces of opposition to illegal immigration is a central theme of his nightly newscasts. Ramos fixates on reports that confirm his conviction with the obsessiveness of an exploitation film director showing close-ups of bullets tearing into human flesh, wrote CIS Fellow Jerry Kammer.He added: On Sunday, Ramos talked with Desierto director Jonas Cuaron and star Gael Garcia Bernal about Trump s repugnant comments about Mexicans. Since then, Ramos said, there are thousands if not millions of North Americans who feel that they have the absolute freedom to be racists. Via: Washington Examiner"
manual_testing(news_article)

# this is a real news
news_article2 = "German Foreign Minister Sigmar Gabriel said it was necessary to do everything possible to make progress on the nuclear deal with Iran and that he did not see any indications during a visit to the United States that Washington would terminate it. U.S. President Donald Trump said earlier on Thursday that “nothing is off the table” in dealing with Iran following its test launch of a ballistic missile."
manual_testing(news_article2)


# Confusion matrix
from sklearn.metrics import confusion_matrix
# Create confusion matrix
cm = confusion_matrix(y_test, pred_dt)

# Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake News', 'Real News'], yticklabels=['Fake News', 'Real News'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
