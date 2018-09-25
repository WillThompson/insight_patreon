# DATA LOADER

from os import listdir
from os.path import isfile, join
import pandas as pd
import re

# Will return a list of files matching a particular regex. Not file paths, just names.
def get_files_matching_regex(path_to_files,file_regex):

    onlyfiles = [f for f in listdir(path_to_files) if isfile(join(path_to_files, f))]
    return([f for f in onlyfiles if bool(re.search(file_regex, f))])

# Assuming all files in a list have the same .csv headers, etc., 
# Returns a pandas dataframe containing all the entries. Duplicate entries are removed as standard
def csvs_to_df(filepaths,drop_duplicates = True):

	if len(filepaths) > 0:
		df = pd.read_csv(filepaths[0])

		for f in filepaths[1:]:
			df = df.append(pd.read_csv(f))

		if drop_duplicates:
			df = df.drop_duplicates()

		return(df)
	else:
		print("WARNING: No filepaths specified.")
		return(pd.DataFrame())

