import data_cleaner
import data_loader
import pickle

# Set the export directory
export_dir = './model/'

# Get the campaign_data
print('Loading data ...')
path_to_data = '../data/'
campaign_regex = 'campaign'
files = data_loader.get_files_matching_regex(path_to_data,campaign_regex)
filepaths = [path_to_data + f for f in files]

campaign_data = data_loader.csvs_to_df(filepaths)

# Also train only on non-empty summaries. Get the indices
notnans = [i for i, x in enumerate(pd.isnull(summaries)) if not x]
summaries = [summaries[x] for x in notnans]

# Get the summaries and clean them
summaries = [data_cleaner.clean_text(str(summ)) for summ in campaign_data.summary]


# Perform the lda analysis

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

num_features = 1000
num_topics = 30

tf_vectorizer = CountVectorizer(
	max_df=0.95, 
	min_df=2, 
	max_features=num_features, 
	stop_words='english')
tf = tf_vectorizer.fit_transform(summaries)
tf_feature_names = tf_vectorizer.get_feature_names()

print('Dumping vectorizer ...')
pickle.dump(tf,open(export_dir + 'lda_vectorizer.pickle','wb'))

print('Creating and training LDA ...')
# Create the LDA object and fit to transformed summaries
lda = LatentDirichletAllocation(
    n_components=num_topics, 
    max_iter=5, 
    learning_method='online', 
    learning_offset=50.,
    random_state=0).fit(tf)

print('Dumping LDA ...')
pickle.dump(lda,open(export_dir + 'lda.pickle','wb'))

#lda_W = lda.transform(tf)
#lda_H = lda.components_

print("Done!")
