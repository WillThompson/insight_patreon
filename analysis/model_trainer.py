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

# Get the campaign_data and the summaries
campaign_data = data_loader.csvs_to_df(filepaths)

# Get the summaries and clean them
summaries = [data_cleaner.clean_text(str(summ)) for summ in campaign_data.summary]
notblanks = [i for i, x in enumerate(pd.isnull(summaries)) if x != '']
summaries = [summaries[x] for x in notblanks]

summaries = summaries[:80000]

# Perform the lda analysis

num_features = 2000
num_topics = 40

print('Creating vectorizer ...')
tf_vectorizer = CountVectorizer(
	max_df=0.95, 
	min_df=2, 
	max_features=num_features, 
	stop_words='english')
tf = tf_vectorizer.fit_transform(summaries)
tf_feature_names = tf_vectorizer.get_feature_names()

print('Dumping vectorizer ...')
pickle.dump(tf_vectorizer,open(export_dir + 'lda_vectorizer.pickle','wb'))

print('Dumping vectorizer tf ...')
pickle.dump(tf,open(export_dir + 'lda_vectorizer_tf.pickle','wb'))

print('Dumping vectorizer tf ...')
pickle.dump(tf_feature_names,open(export_dir + 'lda_vectorizer_tf_fn.pickle','wb'))

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
