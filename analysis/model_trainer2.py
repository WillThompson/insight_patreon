import data_cleaner
import data_loader
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Set the export directory
export_dir = './model/'

# Get the pre-processed data
print('Loading data ...')
data = pd.read_csv('preprocessed_data.csv')

# Get the data that is useful to me.
# i.e. Where the earnings are public and there is a summary of the project
dat = data[(data.earnings_visibility == 'public') & (pd.isnull(data.summary) == False)]

# Load the topic classifier
print('Loading topic classifier from first model trainer ...')

import topic_classifier
tc = topic_classifier.topic_classifier()

# Add the topic classification vectors to the dataframe
print('joining topic classifications ...')

topic_probabilities = tc.get_topic_probs(dat.summary)
topProb = pd.DataFrame(topic_probabilities,columns=['topic'+str(k) for k in range(0,len(topic_probabilities[0]))])
topProb['campaign_id'] = dat.index
topProb = topProb.set_index('campaign_id')
dat = dat.join(topProb)


# Define a crude metric of success
# Assign the labels for success as you see them
dat['success_class'] = 0
dat.loc[dat['pledge_sum'] > 100,'success_class'] = 1
dat.loc[dat['pledge_sum'] > 1000,'success_class'] = 2
dat.loc[dat['pledge_sum'] > 10000,'success_class'] = 3
dat.loc[dat['pledge_sum'] > 100000,'success_class'] = 4

# Separate the data into a training and test set
dat['is_train'] = np.random.uniform(0, 1, len(dat)) <= .80
train, test = dat[dat['is_train']==True], dat[dat['is_train']==False]

# Define which variables to train the random forest on.
cols = [2] + [8,9,10,11] + list(range(23,68))
features = dat.columns[cols]

# Create a random forest Classifier. By convention, clf means 'Classifier'
# Train the Classifier.
print('training classifier...')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)

y = pd.factorize(train['success_class'],sort = True)
clf.fit(train[features], y[0])
print('Dumping classifier ...')
pickle.dump(clf,open(export_dir + 'clf.pickle','wb'))

print('Done!')


