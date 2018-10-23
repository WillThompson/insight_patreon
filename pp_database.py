# Connect to the database

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
import pandas as pd

DB_CONFIG_DICT = {
        'user': 'williamthompson',
        'password': 'passcode',
        'host': 'localhost',
        'port': 5432,
}

DB_CONN_FORMAT = "postgresql://{user}:{password}@{host}:{port}/{database}"
DB_CONN_URI_DEFAULT = (DB_CONN_FORMAT.format(database='patreon_pro_db',**DB_CONFIG_DICT))


# The data base has 5 tables:
#['pp_topic_projections', 'rewards', 'goals', 'campaigns', 'curators']

# Create a connection to the database
def db_connect():
    return create_engine(DB_CONN_URI_DEFAULT)

# Get the curator and campaign information from the postgres database
def get_campaign_info(campaign_id):

	# Write the query and EXECUTE
	cmd = 'SELECT * FROM campaigns JOIN curators ON campaigns.campaign_id = curators.campaign_id WHERE campaigns.campaign_id = {};'
	return(db_connect().execute(cmd.format(campaign_id)))

# Return a pandas DataFrame with all the topics weights as well as other campaign information
def get_random_sample(n_samp):

	# Write the query and EXECUTE
	table1 = 'SELECT campaign_id, campaigns.creation_name, curators.full_name AS curator, curators.youtube, curators.twitter, campaigns.is_monthly, campaigns.creation_count, campaigns.is_nsfw, campaigns.is_charged_immediately, campaigns.pledge_sum FROM campaigns JOIN curators USING (campaign_id) WHERE pledge_sum IS NOT NULL LIMIT {}'.format(n_samp)
	table2 = 'SELECT campaign_id, reward_count, COALESCE(goal_count,0) AS goal_count FROM (SELECT campaign_id, COUNT(campaign_id)-2 AS reward_count FROM rewards GROUP BY campaign_id) AS baz FULL OUTER JOIN (SELECT campaign_id, COUNT(campaign_id) AS goal_count FROM goals GROUP BY campaign_id) AS bar USING (campaign_id)'
	table3 = 'SELECT * FROM ({}) AS bar1 JOIN ({}) AS bar2 USING (campaign_id)'.format(table1,table2)
	cmd = 'SELECT * FROM ({}) AS foo JOIN pp_topic_projections USING (campaign_id)'.format(table3)

	db = pd.read_sql(cmd+';',db_connect())

	db['curatorHasTwitter'] = pd.isnull(db['twitter'])
	db['curatorHasYoutube'] = pd.isnull(db['youtube'])

	db.pop('twitter')
	db.pop('youtube')

	return(db)







