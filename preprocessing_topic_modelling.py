# Perform the topic_modelling preprocessing
import pandas as pd

# Load data
print('loading data ...')

data = pd.read_csv('preprocessed_data.csv')
data = data[~pd.isnull(data.summary)].set_index('campaign_id')

print('creating topic classifier ...')

import analysis.topic_classifier
tc = analysis.topic_classifier.topic_classifier(
	'model/lda_TEMP.pickle',
	'model/lda_vectorizer_TEMP.pickle',
	'model/lda_vectorizer_tf_TEMP.pickle')

print('pre-allocating space ...')

#Pre-allocate space
num_topics = tc.lda_model.n_components
num_summaries = len(data)

topics = [[0]*num_topics]*num_summaries	


print('processing summaries for topic info. ...')

for k in range(0,num_summaries):
	topics[k] = tc.get_topic_probs([data.iloc[k].summary])

print('writing topic data to file ...')

# Create a dataframe of the topic vectorizations

topic_names = ["topic"+str(k) for k in range(0,num_topics)]
df = pd.DataFrame(topics, columns=topic_names).set_index(data.index)
data = data.join(df)

data.to_csv('preprocessed_data_topics_TEMP.csv')

print('Done.')