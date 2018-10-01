# Module for performing some basic analysis of similar campaigns

# Import the plotting libraries

from bokeh.plotting import figure
from bokeh.embed import components

import pandas as pd
import numpy as np

# Make a plot of the pledge sums for camplaigns in comparison data
def make_pledge_histogram(comparison_data):

	plot = figure(plot_height=300, sizing_mode='scale_width',background_fill_color="#FFFFFF",x_axis_type='log',tools=[])
	idx = (comparison_data.earnings_visibility == 'public') & (comparison_data.pledge_sum > 0)
	y = comparison_data[idx]['pledge_sum']

	# Create the histogram
	hist, edges = np.histogram(y/100, density=False, bins = [10**(x/4) for x in range(-8, 20)])

	plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#ff69b4", line_color="#ff69b4")

	plot.yaxis.axis_label = 'Number of campaigns'
	plot.xaxis.axis_label = 'Pledge Sum (US$)'

	return(plot)

# Make a plot of the pledge sums for camplaigns in comparison data
def make_patron_vs_pledge_scatter(comparison_data,entry):

	plot = figure(plot_height=300, sizing_mode='scale_width',background_fill_color="#FFFFFF",x_axis_type='log',y_axis_type='log',tools=[])
	idx = (comparison_data.earnings_visibility == 'public') & (comparison_data.pledge_sum > 0) & (comparison_data.patron_count > 0)
	
	x = comparison_data[idx]['patron_count']
	y = comparison_data[idx]['pledge_sum']


	colours = ["#%02x%02x%02x" % (int(150*x),int(150*x),int(150*x)) for x in comparison_data[idx]['distance']]

	plot.scatter(x,y, fill_color=colours, line_color=colours)
	
	#if the campaign has visible pledge data, then display their campaign against the others.
	if entry.iloc[0]['earnings_visibility'] == 'public':
		xx = entry['patron_count']
		yy = entry['pledge_sum']
		plot.scatter(xx,yy, fill_color="#ff69b4", line_color="#ff69b4")


	plot.xaxis.axis_label = 'Number of Patrons'
	plot.yaxis.axis_label = 'Pledge Sum (US$)'

	# Remove the logo and toolbar
	plot.toolbar.logo = None
	plot.toolbar_location = None
	
	return(plot)


def get_basic_statistics(comparison_data):

	base_stats = {}
	base_stats['mean_pledge_sum'] = np.mean(comparison_data['pledge_sum'])

	return(base_stats)

# create the code to embed the plots into html.
def render_for_html(plot):

	script, div = components(plot)
	return script, div