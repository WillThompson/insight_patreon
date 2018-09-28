from applet_new.patreonpro import app

from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, g, flash
import datetime
import pandas as pd
import numpy as np
import patreon_requests
import analysis.topic_classifier
import tokenizer
import scipy.stats as sp
import pickle

def days_since_creation(dates):
    now = datetime.datetime.now()
    date_diffs = [datetime.datetime.strptime(x[0:10],"%Y-%m-%d") for x in dates]
    return([(now - d).days for d in date_diffs])

## ROUTING
@app.route('/',methods=['GET','POST'])
def home():

	if request.method == 'POST':
		
		campaign_url = request.form['campaign_url']
		campaign_id = patreon_requests.get_campaign_id_from_url(campaign_url)
		return redirect(url_for('render_prediction',campaign_id=campaign_id))

	return render_template('home.html')

# HELPER
def jensen_shannon_distance(v1,v2):
    
    m = 0.5*(v1 + v2)
    U = sp.entropy(v1, m, base=None)
    V = sp.entropy(v2, m, base=None)
    return(0.5*(U + V))


@app.route('/<campaign_id>')
def render_prediction(campaign_id):
	
	cur,cpn,rwd = patreon_requests.get_campaign_info_from_id(campaign_id)
	entry = pd.DataFrame(cpn)

	if(pd.isnull(entry.iloc[0].summary)):
		flash('No summary included in campaign. Not enough information to group.')
		return redirect(url_for('home'))

	# Get the title of the campaign
	title = entry.iloc[0]['creation_name']

	# Compute some extra variables needed for later
	entry['creation_rate'] = entry['creation_count']/days_since_creation(entry['created_at'])
	entry['curatorHasYoutube'] = ~pd.isnull(cur.youtube)
	entry['curatorHasTwitter'] = ~pd.isnull(cur.twitter)

	goal_count = len(rwd[rwd['type'] == 'goal'])
	entry['goal_count'] = goal_count
	reward_count = len(rwd[rwd['type'] == 'reward']) - 2
	entry['reward_count'] = reward_count

	## Pull the summary for analysis
	summary = entry.iloc[0].summary
	prepared_text = tokenizer.prepare_text_for_lda(summary)

	tc = analysis.topic_classifier.topic_classifier()
	top_topics = tc.get_most_likely_topics(summary,3,display=False)	
	topic_probs = tc.get_topic_probs(summary)

	topic_prob_df = pd.DataFrame([{'topic'+str(k): topic_probs[k] for k in range(0,len(topic_probs))}]).set_index(entry.index)
	e = entry.join(topic_prob_df)

	# Load a dataframe with a sample set of campaigns that can be used to compare to
	other_camps = pd.read_csv('sample_data.csv')
	
	dd = []
	for k in range(0,1000):
		vv = tc.get_topic_probs(other_camps.iloc[k].summary)
		dd.append(jensen_shannon_distance(topic_probs,vv))

	sorted_inds = np.argsort(dd)

	prediction_inds = [x for x in sorted_inds[0:3]]
	prediction_topics = [tc.get_most_likely_topics(other_camps.iloc[x].summary,3,display=False) for x in prediction_inds]

	return render_template('prediction.html', campaign_id=campaign_id, title=title, my_new_var=summary, top_topics=top_topics, prediction=prediction_topics)
