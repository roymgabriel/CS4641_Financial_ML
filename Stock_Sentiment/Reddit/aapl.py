# Import required libraries
import nest_asyncio
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import datetime

# Yahoo API
import yfinance as yf

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# This ensures the twint loop can run in jupyter notebook, no need to understand it
nest_asyncio.apply()

##################################################################################


vadf = pd.read_csv("1421_aapl.csv")


vadf = vadf.sort_values(by = ["Date"], ascending=False)


x = vadf.drop(['AAPL_pct_change',"Title"], axis=1)
y = vadf[['AAPL_pct_change']]


for i, word in enumerate(y.values):
    if word == 'BULLISH':
        y.values[i] = 1
    elif word == "BEARISH":
        y.values[i] = 0
    else:
        raise ValueError
        
y = y.astype('float')

for i, j in enumerate(vadf['Date'].values):
    if j.startswith("2019"):
        break

x_test = x.iloc[:i, 1:]
y_test = y.iloc[:i, :]

x_train = x.iloc[i:, 1:]
y_train = y.iloc[i:, :]

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

export = pd.concat([vadf.iloc[:i, :]['Date'], y_test] , axis = 1)
export = export.reset_index(drop=True)


sc_Up = StandardScaler()
sc_Down = StandardScaler()
sc_CountComments = StandardScaler()
sc_Negative_Score = StandardScaler()
sc_Neutral_Score = StandardScaler()
sc_Positive_Score = StandardScaler()
sc_y = StandardScaler() # pct_change

sc_Award = StandardScaler()
sc_Uprate = StandardScaler()
sc_Crosspost = StandardScaler()

#print(x_train.Up.values.reshape(-1,1))
# training
standard_Up_train = sc_Up.fit_transform(x_train.Up.values.reshape(-1,1))
standard_Down_train = sc_Down.fit_transform(x_train.Down.values.reshape(-1,1))
standard_CountComments_train= sc_CountComments.fit_transform(x_train.CountComments.values.reshape(-1,1))
standard_Negative_Score_train = sc_Negative_Score.fit_transform(x_train.Negative_Score.values.reshape(-1,1))
standard_Neutral_Score_train = sc_Neutral_Score.fit_transform(x_train.Neutral_Score.values.reshape(-1,1))
standard_Positive_Score_train = sc_Positive_Score.fit_transform(x_train.Positive_Score.values.reshape(-1,1))

standard_Award_train = sc_Award.fit_transform(x_train.Award.values.reshape(-1,1))
standard_Uprate_train = sc_Uprate.fit_transform(x_train.Uprate.values.reshape(-1,1))
standard_Crosspost_train = sc_Crosspost.fit_transform(x_train.Crosspost.values.reshape(-1,1))

# testing
standard_Up_test = sc_Up.transform(x_test.Up.values.reshape(-1,1))
standard_Down_test = sc_Down.transform(x_test.Down.values.reshape(-1,1))
standard_CountComments_test = sc_CountComments.transform(x_test.CountComments.values.reshape(-1,1))
standard_Negative_Score_test = sc_Negative_Score.transform(x_test.Negative_Score.values.reshape(-1,1))
standard_Neutral_Score_test = sc_Neutral_Score.transform(x_test.Neutral_Score.values.reshape(-1,1))
standard_Positive_Score_test = sc_Positive_Score.transform(x_test.Positive_Score.values.reshape(-1,1))

standard_Award_test = sc_Award.transform(x_test.Award.values.reshape(-1,1))
standard_Uprate_test = sc_Uprate.transform(x_test.Uprate.values.reshape(-1,1))
standard_Crosspost_test = sc_Crosspost.transform(x_test.Crosspost.values.reshape(-1,1))

# training
standard_y_train = sc_y.fit_transform(y_train.AAPL_pct_change.values.reshape(-1,1)) # pct_change

# testing
standard_y_test = sc_y.fit_transform(y_test.AAPL_pct_change.values.reshape(-1,1)) # pct_change



# in case you want to visualize standardized training data
# make a copy of original df
df_training = x_train.copy()

# Independent Variables"
df_training['Up'] = standard_Up_train
df_training['Down'] = standard_Down_train
df_training['CountComments'] = standard_CountComments_train
df_training['Negative_Score'] = standard_Negative_Score_train
df_training['Neutral_Score'] = standard_Neutral_Score_train
df_training['Positive_Score'] = standard_Positive_Score_train

df_training['Award'] = standard_Award_train
df_training['Uprate'] = standard_Uprate_train
df_training['Crosspost'] = standard_Crosspost_train

# Dependant Variable
df_training['AAPL_pct_change'] = standard_y_train


# in case you want to visualize standardized testing data
# make a copy of original df
df_testing = x_test.copy()

# Independent Variables"
df_testing['Up'] = standard_Up_test
df_testing['Down'] = standard_Down_test
df_testing['CountComments'] = standard_CountComments_test
df_testing['Negative_Score'] = standard_Negative_Score_test
df_testing['Neutral_Score'] = standard_Neutral_Score_test
df_testing['Positive_Score'] = standard_Positive_Score_test

df_testing['Award'] = standard_Award_test
df_testing['Uprate'] = standard_Uprate_test
df_testing['Crosspost'] = standard_Crosspost_test

# Dependant Variable
df_testing['AAPL_pct_change'] = standard_y_test


x_test = df_testing.iloc[:, :-1]
x_train = df_training.iloc[:, :-1]

x_test = x_test.drop('Date', axis=1)
x_train = x_train.drop('Date', axis=1)


##################################################################################NB Ber 50

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(x_train.values, y_train.values.ravel())

# predict 
y_predicted = classifier.predict(x_test.values) 

#plot
print(f"Classification Report for Naive Bayes\n\n")
print(classification_report(y_test.values, y_predicted, target_names=['Bullish', 'Bearish']))
plot_confusion_matrix(classifier, x_test.values, y_test.values, cmap=plt.cm.Blues, display_labels=['Bullish', 'Bearish'])  
plt.show()



export['NB_Ber_Predictions'] = pd.Series(y_predicted)


##################################################################################Dec 50

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train.values, y_train.values.ravel())

# predict 
y_predicted = classifier.predict(x_test.values) 

#plot
print(f"Classification Report for Decision Tree Classification\n\n")
print(classification_report(y_test.values, y_predicted, target_names=['Bullish', 'Bearish']))
plot_confusion_matrix(classifier, x_test.values, y_test.values, cmap=plt.cm.Blues, display_labels=['Bullish', 'Bearish'])  
plt.show()



export['Decision_Tree_Predictions'] = pd.Series(y_predicted)

##################################################################################

export["Stock"] = "AAPL"

export.to_csv("nn_aapl.csv")

print("Finish Export~~")

