# Load the training model
import pickle

class success_classifier:

	def __init__(self):

		self.classifier = pickle.load(open('patreonpro/model/clf.pickle','rb'))
		self.features = ['creation_count', 'is_monthly', 'is_charged_immediately',
       'display_patron_goals', 'topic0', 'topic1', 'topic2', 'topic3',
       'topic4', 'topic5', 'topic6', 'topic7', 'topic8', 'topic9', 'topic10',
       'topic11', 'topic12', 'topic13', 'topic14', 'topic15', 'topic16',
       'topic17', 'topic18', 'topic19', 'topic20', 'topic21', 'topic22',
       'topic23', 'topic24', 'topic25', 'topic26', 'topic27', 'topic28',
       'topic29', 'topic30', 'topic31', 'topic32', 'topic33', 'topic34',
       'topic35', 'topic36', 'topic37', 'topic38', 'topic39']
	# Return a dictionary of the topic probabilities
	def assess_success(self,db_entry):
		return self.classifier.predict_proba(db_entry[self.features])[0][1]






