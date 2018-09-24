import os
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, g, flash
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
from subprocess import call
import datetime
import numpy.random as rand

import patreonpro.topic_classifier
import patreonpro.success_classifier
import patreonpro.patreon_requests
import pandas as pd
import pickle
import patreonpro.data_cleaner as datcle

DEBUG=True
app = Flask(__name__)
app.config.from_object(__name__)
app.config.update(dict(
	SECRET_KEY = 'there_is_no_secret_key',
	USERNAME='admin',
    PASSWORD='default'
    ))


# Load the dataframe of campaigns as a global variable.

## ROUTING
@app.route('/',methods=['GET','POST'])
def home():

	if request.method == 'POST':
		
		campaign_url = request.form['campaign_url']
		campaign_id = patreonpro.patreon_requests.get_campaign_id_from_url(campaign_url)
		return redirect(url_for('render_prediction',campaign_id=campaign_id))

	return render_template('home.html')

@app.route('/prediction/<campaign_id>')
def render_prediction(campaign_id):
	#df = pd.read_csv('../data/campaigns_1537480072.csv')
	#entry = pd.DataFrame([df.iloc[int(campaign_id)]]).set_index('campaign_id')
	#title = entry.iloc[0]['creation_name']
	# If there is no summary for the campaign, then ignore it ... for now.

	entry = pd.DataFrame(patreonpro.patreon_requests.get_campaign_info_from_id(campaign_id))

	if(pd.isnull(entry.iloc[0].summary)):
		flash('No summary included in campaign. Please choose another campaign.')
		return redirect(url_for('home'))

	title = entry.iloc[0]['creation_name']

	# Import the classifiers, clean the text of the summary according to our rules.
	tc = patreonpro.topic_classifier.topic_classifier()
	sc = patreonpro.success_classifier.success_classifier()
	summ = datcle.clean_text(entry.iloc[0].summary)

	top_topics = tc.get_most_likely_topics(summ,3,display=False)	
	topic_probabilities = tc.get_topic_probs(summ)
	topic_probs = tc.get_topic_probs(entry.summary)

	topProb = pd.DataFrame(topic_probs,columns=['topic'+str(k) for k in range(0,len(topic_probs[0]))]).set_index(entry.index)
	e = entry.join(topProb)
	pred = sc.assess_success(e)
	pred_string = '{:.0%} chance of "success".'.format(pred)

	return render_template('prediction.html', campaign_id=campaign_id, title=title, my_new_var=summ, top_topics=top_topics,prediction=pred_string)

## -- MAIN -- ##
if __name__ == "__main__":
    app.run(debug=DEBUG, use_reloader=True)


