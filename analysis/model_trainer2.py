import data_cleaner
import data_loader
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Set the export directory
export_dir = './model/'

# Get the campaign_data
print('Loading data ...')
path_to_data = '../data/'
campaign_regex = 'campaign'
files = data_loader.get_files_matching_regex(path_to_data,campaign_regex)
filepaths = [path_to_data + f for f in files]

campaign_data = data_loader.csvs_to_df(filepaths)
campaign_data = campaign_data.set_index('campaign_id')

reduced_data = campaign_data[['created_at','published_at','creation_count','is_monthly','is_charged_immediately','display_patron_goals','earnings_visibility','summary','pledge_sum','patron_count']]

# Define a crude metric of success
reduced_data['success'] = (reduced_data['patron_count'] > 30) & (reduced_data['pledge_sum'] > 3000)

dat = reduced_data[(reduced_data.earnings_visibility == 'public') & (pd.isnull(reduced_data.summary) == False)]

from sklearn.ensemble import RandomForestClassifier
import topic_classifier

print('Loading topic classifier from first model trainer ...')

tc = topic_classifier.topic_classifier()

dat = reduced_data[(reduced_data.earnings_visibility == 'public') & (pd.isnull(reduced_data.summary) == False)]
topic_probabilities = tc.get_topic_probs(dat.summary)
topProb = pd.DataFrame(topic_probabilities,columns=['topic'+str(k) for k in range(0,len(topic_probabilities[0]))])
topProb['campaign_id'] = dat.index
topProb = topProb.set_index('campaign_id')
dat = dat.join(topProb)
e = dat.pop('earnings_visibility')

dat['is_train'] = np.random.uniform(0, 1, len(dat)) <= .75
train, test = dat[dat['is_train']==True], dat[dat['is_train']==False]

print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

cols = [2,3,4,5]+list(range(10,50))
features = dat.columns[cols]

y = pd.factorize(train['success'])[0]
y = 1 - y

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier.
clf.fit(train[features], y)
print('Dumping classifier ...')
pickle.dump(clf,open(export_dir + 'clf.pickle','wb'))


