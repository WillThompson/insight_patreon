# Load the training model
import pickle

class topic_classifier:

	def __init__(self,lda_model_path,vectorizer_path,tf_path):

		# Load the appropriate pickle files from previous training.
		lda = pickle.load(open(lda_model_path,'rb'))
		vectorizer = pickle.load(open(vectorizer_path,'rb'))
		tf = pickle.load(open(tf_path,'rb'))
		tf_feature_names = vectorizer.get_feature_names()

		self.lda_model = lda
		self.vectorizer = vectorizer

		def get_topics(model, feature_names, no_top_words):
		    ll = list()
		    for topic_idx, topic in enumerate(model.components_): 
		        st = ", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
		        ll += [st]
		    return(ll)

		self.topics = get_topics(self.lda_model, tf_feature_names, 10)

	# Return a dictionary of the topic probabilities
	def get_topic_probs(self,text):

		if type(text) == str:
			text = [text]

		mapped_summary = self.vectorizer.transform(text)
		topic_probs = self.lda_model.transform(mapped_summary)

		return(topic_probs[0])

	def get_topic_probs_dictionary(self,text):

		topic_probs = self.get_topic_probs(text)
		tp_dict = {'topic'+str(k):topic_probs[k] for k in range(0,len(topic_probs))}
		return(tp_dict)

	def get_most_likely_topics(self,text,num_most_likely,display=True):

		top = [[i,j] for (i,j) in zip(self.get_topic_probs(text),self.topics)]
		q = sorted(top,reverse=True)
		#if display:
		#	for k in range(0,num_most_likely):
		#		print('{:.2%}: {}'.format(*list(q[k])))

		return(q[:num_most_likely])

	def get_topics(self):
		return self.topics




