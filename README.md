import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

df = pd.read_csv('/content/spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

# Replace ham with 0 and spam with 1
df = df.replace(['ham','spam'],[0, 1])

df.head()

df.columns

df['Count']=0
for i in np.arange(0,len(df.v2)):
    df.loc[i,'Count'] = len(df.loc[i,'v2'])

df.head()

# Total ham(0) and spam(1) messages
df['v1'].value_counts()

df.info()

corpus = []
ps = PorterStemmer()

print (df['v2'][0])
print (df['v2'][1])

nltk.download('stopwords')

for i in range(0, 5572):

    # Applying Regular Expression

    '''import nltk

    Replace email addresses with 'emailaddr'
    Replace URLs with 'httpaddr'
…
    # Preparing WordVector Corpus
    corpus.append(msg)

cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()

y = df['v1']
print (y.value_counts())

print(y[0])
print(y[1])

le= LabelEncoder()
y = le.fit_transform(y)

print(y[0])
print(y[1])

xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size= 0.20, random_state = 0)

bayes_classifier = GaussianNB()
bayes_classifier.fit(xtrain, ytrain)

# Predicting
y_pred = bayes_classifier.predict(xtest)

cm = confusion_matrix(ytest, y_pred)

cm

print ("Accuracy : %0.5f \n\n" % accuracy_score(ytest, bayes_classifier.predict(xtest)))
print (classification_report(ytest, bayes_classifier.predict(xtest)))

dt = DecisionTreeClassifier(random_state=50)
dt.fit(xtrain, ytrain)

# Predicting
y_pred_dt = dt.predict(xtest)
























