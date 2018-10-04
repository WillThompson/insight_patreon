# Use this file to define the preprocessing methods for all the data

import data_loader
import data_cleaner
import datetime
import pandas as pd

def days_since_creation(dates):
    now = datetime.datetime.now()
    date_diffs = [datetime.datetime.strptime(x[0:10],"%Y-%m-%d") for x in dates]
    return([(now - d).days for d in date_diffs])

def main():

	# Load ALL the data into the campaigns DataFrame
	print('loading data files ...')
	path_to_files = './data/'
	cmpn_reg = 'campaigns'
	crtr_reg = 'curators'
	rwrd_reg = 'rewards'
	campaigns = data_loader.csvs_to_df([path_to_files + f for f in data_loader.get_files_matching_regex(path_to_files,cmpn_reg)])
	curators = data_loader.csvs_to_df([path_to_files + f for f in data_loader.get_files_matching_regex(path_to_files,crtr_reg)])
	rewards = data_loader.csvs_to_df([path_to_files + f for f in data_loader.get_files_matching_regex(path_to_files,rwrd_reg)])

	print('creating dataframe ...')

	# Campaign data is loaded. Now it needs to be cleaned and preprocessed for the analysis stage
	campaigns['creation_rate'] = campaigns['creation_count']/days_since_creation(campaigns['created_at'])

	campaigns['summary'].loc[pd.isnull(campaigns['summary'])] = ''
	#campaigns['summary'] = [data_cleaner.clean_text(x) for x in campaigns['summary']]

	# Get the campaign dataframe as a start
	df = campaigns.set_index('campaign_id')

	# Get the appropriate information from the curator page and add it to the df
	df_temp = pd.DataFrame(curators['campaign_id'])
	df_temp['curator'] = curators.full_name
	df_temp['curatorHasYoutube'] = ~pd.isnull(curators.youtube)
	df_temp['curatorHasTwitter'] = ~pd.isnull(curators.twitter)
	df_temp = df_temp.set_index('campaign_id')

	# join it up
	df = df.join(df_temp)

	# Get the appropriate information from the rewards page and add it to the df
	df_temp = pd.DataFrame(curators['campaign_id'])
	df_temp = df_temp.set_index('campaign_id')

	goal_count = rewards[rewards['type'] == 'goal'].groupby(['campaign_id'])['type'].count()
	reward_count = rewards[rewards['type'] == 'reward'].groupby(['campaign_id'])['type'].count() - 2
	df_temp['goal_count'] = goal_count
	df_temp['reward_count'] = reward_count
	df_temp = df_temp.fillna(0)

	# join it up
	df = df.join(df_temp)

	print('writing dataframe to file ...')
	df.to_csv('preprocessed_data.csv')

	print('preprocessing complete.')



if __name__ == '__main__':
	main()


