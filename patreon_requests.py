
from bs4 import BeautifulSoup
import requests
import json
import pandas as pd

# Get the campaign ID from the url of the patreon page.
def get_campaign_id_from_url(url):

	response = requests.get(url)

	# If the response is good, do something:
	html = response.content
	parsed_html = BeautifulSoup(html)
	x = parsed_html.head.find('link', attrs={'rel':"image_src"}).get('content')
	xx = x.split('/')

	return(int(xx[xx.index('campaign') + 1]))

# Query the Patreon API to get the campaign information
def get_campaign_info_from_id(campaign_id):

	url = "https://api.patreon.com/campaigns/{}".format(str(campaign_id))
	response = requests.get(url)

	# if the response is good, do something
	dictionary = json.loads(response.content)
	cur,cpn,rwd = get_parsed_data(dictionary)

	return(cur,cpn,rwd)

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