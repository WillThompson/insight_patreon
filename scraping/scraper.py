import requests
import json
import pandas as pd
import sys

# Return a dataframe row that can be inserted into an existing dataframe
def get_parsed_data(dic):

	campaign = dict(dic['data']['attributes'])
	campaign = parse_campaign(campaign)

	curator = dict(dic['included'][0])
	curator = parse_curator(curator)

	tiers = dic['included'][1:]
	tiers = parse_tiers(tiers)
	
	return(curator,campaign,tiers)

# Get all the relevant information about the tiers for a campaign. Return a dataframe object.
def parse_tiers(dic):
	
	tiers = pd.DataFrame()
	for tier in dic:
		t = pd.DataFrame([tier['attributes']])
		t['id'] = tier['id']
		t['type'] = tier['type']    
		tiers = tiers.append(t)      
	return(tiers)

# Get all the relevant information about the campaign. Return a dataframe object.
def parse_campaign(dic):
	campaign = pd.DataFrame([dic])
	return(campaign)

# Get all the relevant information about the curator for a campaign. Return a dataframe object.
def parse_curator(dic):
	
	# get attributes
	attrib = dic['attributes']
	social_connections = attrib.pop('social_connections')
	social_connections = {'social_'+d:social_connections[d] for d in social_connections}
	attrib.update(social_connections)
	
	curator = pd.DataFrame([attrib])
	return(curator)

def main():

	# Set the parameters for your call to the API
	request_base = "https://api.patreon.com/campaigns/{}"
	range_min = 50000
	diff = 50000

	curators = pd.DataFrame()
	campaigns = pd.DataFrame()
	rewards = pd.DataFrame()

	for k in range(range_min,range_min + diff):

		url = request_base.format(str(k))
		myResponse = requests.get(url)
		if(myResponse.ok):
			# Covert the .json data to a dictionary.
			dictionary = json.loads(myResponse.content)
			
			# handle data that has been downloaded.
			cur,cpn,rwd = get_parsed_data(dictionary)
			
			# append the data to the dataframe
			curators = curators.append([cur])
			campaigns = campaigns.append([cpn])
			rewards = rewards.append([rwd])

		sys.stdout.write("{0:.2%} complete.\r".format((k-range_min)/diff))
		sys.stdout.flush()

		for data,filename in zip([curators,campaigns,rewards],['curators','campaigns','rewards']):
			data.to_csv(filename+'.csv',index=False)

	sys.stdout.write("Done.              \n")


if __name__ == '__main__':
	main()