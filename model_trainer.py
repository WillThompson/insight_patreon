import data_cleaner
import data_loader
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Set the export directory
export_dir = './model/'


# Load the preprocessed campaign data
print('loading data ...')
data = pd.read_csv('./analysis/preprocessed_data.csv')

# Get the summaries and clean them
summs = data[~pd.isnull(data.summary)].summary

# Take only the first few thousand summaries
summaries = summs[:20000]

# import the code to generate the custom tokenizer
import tokenizer

num_features = 40000

print('creating vectorizer and vectorizing data...')

vectorizer = TfidfVectorizer(
	tokenizer=tokenizer.prepare_text_for_lda, 
	ngram_range=(1,1), 
	max_df=0.3, 
	min_df=0.01, 
	max_features=num_features,
	stop_words='english')

tf = vectorizer.fit_transform(summaries)

print('Dumping vectorizer ...')
pickle.dump(vectorizer,open(export_dir + 'lda_vectorizer.pickle','wb'))

print('Dumping vectorizer fit ...')
pickle.dump(tf,open(export_dir + 'lda_vectorizer_tf.pickle','wb'))


print('Creating and training LDA ...')
# Create the LDA object and fit to transformed summaries

num_topics = 15

lda = LatentDirichletAllocation(
    n_components=num_topics, 
    max_iter=20, 
    learning_method='batch', 
    learning_offset=50.,
    evaluate_every=2,
    verbose=True,
    random_state=0).fit(tf)

print('Dumping LDA ...')
pickle.dump(lda,open(export_dir + 'lda.pickle','wb'))

print("Done!")
