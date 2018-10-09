from applet.patreonpro import app

from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, g, flash
import datetime
import pandas as pd
import numpy as np
import patreon_requests
import analysis.topic_classifier
import analysis.data_object
import analysis.plots
import analysis.pp_stats
import tokenizer
import scipy.stats as sp


print('preparing topic classifier')
tc = analysis.topic_classifier.topic_classifier(
	'model/lda.pickle',
	'model/lda_vectorizer.pickle',
	'model/lda_vectorizer_tf.pickle')

#print('loading pre-processed data')
#ppdata = analysis.data_object.data_object(
#	'preprocessed_data_topics.csv')

## ROUTING
@app.route('/',methods=['GET','POST'])
def home():

	if request.method == 'POST':
		
		campaign_url = request.form['campaign_url']
		campaign_id = patreon_requests.get_campaign_id_from_url(campaign_url)
		return redirect(url_for('render_prediction',campaign_id=campaign_id))

	return render_template('home.html')


@app.route('/<campaign_id>')
def render_prediction(campaign_id):

	# Query PATREON for information on the campaign
	print('Calling PATREON for information on campaign {} ...'.format(campaign_id))
	cur,cpn,rwd = patreon_requests.get_campaign_info_from_id(campaign_id)
	entry = pd.DataFrame(cpn)

	# If the campaign has no summary, then do not perform the comparison...
	if(pd.isnull(entry.iloc[0].summary)):
		flash('No summary included in campaign. Not enough information to group.')
		return redirect(url_for('home'))

	# Add some extra variables needed for later
	entry['curator'] = cur.iloc[0]['full_name']
	entry['curatorHasYoutube'] = ~pd.isnull(cur.youtube)
	entry['curatorHasTwitter'] = ~pd.isnull(cur.twitter)

	goal_count = len(rwd[rwd['type'] == 'goal'])
	entry['goal_count'] = goal_count
	reward_count = len(rwd[rwd['type'] == 'reward']) - 2
	entry['reward_count'] = reward_count

	## Pull the summary for analysis
	print('preparing campaign {} summary text for LDA ...'.format(campaign_id))

	## Tokenize the text
	prepared_text = tokenizer.prepare_text_for_lda(entry.iloc[0]['summary'])
	top_topics = tc.get_most_likely_topics(entry.iloc[0]['summary'],3,display=False)	
	topic_probs = tc.get_topic_probs(entry.iloc[0]['summary'])

	n_topics = len(topic_probs)
	topic_labels = ['topic'+str(k) for k in range(0,n_topics)]
	topic_prob_df = pd.DataFrame([{'topic'+str(k): topic_probs[k] for k in range(0,n_topics)}]).set_index(entry.index)
	e = entry.join(topic_prob_df)

	# Load a dataframe with a sample set of campaigns that can be used to compare to



	num_campaigns = 291049
	n_comp = 100000
	#all_campaigns = ppdata.get_dataframe()
	#inds = np.random.choice(len(all_campaigns), n_comp, replace=False)

	# Grab a subset of the campaigns for now....
	print('grabbing {} campaigns for comparison ...'.format(n_comp))
	inds = sorted(np.random.choice(np.arange(1,num_campaigns+1), num_campaigns - n_comp, replace=False))
	comparison = pd.read_csv('preprocessed_data_topics.csv',skiprows=inds)

	# Get the target probability distribution for the topics
	print('getting sorted indices ...')
	target = topic_probs
	sorted_inds = analysis.pp_stats.get_sorted_indices_of_closest(target,comparison[topic_labels])
	prediction_inds = [s[1] for s in sorted_inds[0:5]]

	# Get the cloest campaigns 'similar campaigns'
	n_compare = 3000
	distances, stats_inds = [p[0] for p in sorted_inds[:n_compare]],[p[1] for p in sorted_inds[:n_compare]]
	similar_campaigns = comparison.iloc[stats_inds]
	similar_campaigns['distance'] = distances

	# Print the similar campaigns
	print(similar_campaigns[['curator','distance']])
	
	campaigns_to_return = similar_campaigns.iloc[0:5].iterrows()
	# Get the significant predictors

	non_nulls = ~pd.isnull(similar_campaigns['pledge_sum'])
	possible_predictors = ['creation_count',
         'curatorHasTwitter',
         'curatorHasYoutube',
         'is_charged_immediately',
         'is_monthly',
         'is_nsfw',
         'goal_count',
         'reward_count']
	target_variable = 'pledge_sum'
	significant_predictors = analysis.pp_stats.get_significant_predictors(similar_campaigns[non_nulls],possible_predictors,[target_variable],significance_level=0.05)

	# Get the statistics of the best v the worst campaigns
	proportion = 0.10

	ranked_campaigns = similar_campaigns.sort_values(['pledge_sum'])
	best_campaigns = ranked_campaigns.tail(int(np.round(proportion*n_compare)))
	worst_campaigns = ranked_campaigns.head(int(np.round(proportion*n_compare)))

	best_v_worst_stats = analysis.pp_stats.best_v_worst_stats(significant_predictors,best_campaigns,worst_campaigns)

	quantile = analysis.pp_stats.get_quantile(similar_campaigns,e)

	recommendations = analysis.pp_stats.get_all_recommendations(significant_predictors,best_campaigns,worst_campaigns,entry)

	# Get the closest of the most successful campaigns to compare to.
	closest_of_the_best = best_campaigns.sort_values(['distance']).iloc[0:5].iterrows()

	return render_template('prediction.html', 
		campaign_info=entry.iloc[0],
		sim_camps = campaigns_to_return,
		closest_of_the_best = closest_of_the_best, 
		top_topics=top_topics,
		significant_predictors=significant_predictors,
		best_v_worst_stats = best_v_worst_stats,
		quantile=quantile,
		recommendations=recommendations
		)
