# Write a command that will perform 
# Latent Dirichlet allocation on a group of campaign text summaries.

# https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730

# Import sci-kit learn libraries.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Import other libraries
import pandas as pd
import re

num_features = 2000
num_topics = 40


tf_vectorizer = TfidfVectorizer(
	max_df=0.95, 
	min_df=2, 
	max_features=num_features, 
	stop_words='english')
tf = tf_vectorizer.fit_transform(summaries)
tf_feature_names = tf_vectorizer.get_feature_names()

# Create the LDA object and fit to transformed summaries
lda = LatentDirichletAllocation(
    n_components=num_topics, 
    max_iter=5, 
    learning_method='online', 
    learning_offset=50.,
    random_state=0).fit(tf)

if __name__ == '__main__':
	#Do nothing

