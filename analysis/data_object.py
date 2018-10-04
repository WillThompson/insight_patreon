import pandas as pd

class data_object:

	def __init__(self,path_to_data_file):

		self.dataframe = pd.read_csv(path_to_data_file)

	def get_dataframe(self):

		return(self.dataframe)
