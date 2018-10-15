import scipy.stats as sp
import sklearn.linear_model
import numpy as np
import pandas as pd

# Compute and return the Jensen-Shannon Distance between two distributions
def jensen_shannon_distance(v1,v2):
    
    m = 0.5*(v1 + v2)
    U = sp.entropy(v1, m, base=None)
    V = sp.entropy(v2, m, base=None)
    return(0.5*(U + V))

# Set the standard distance measure to be used in calculations.
def get_distance_between_dists(v1,v2):
	return(jensen_shannon_distance(v1,v2))


# Inputs: The target distribution. The set of data to compare to.
# Outputs: The indices of the data that are 'closest' to the target
def get_sorted_indices_of_closest(target,comparison):

	#pre-allocate array
	dists_and_inds = [(0.0,0)]*len(comparison)

	# Convert to a numpy array
	v1 = np.array(target).astype('float')

	for k in range(0,len(comparison)):

		v2 = np.array(comparison.iloc[k]).astype('float')
		dists_and_inds[k] = (get_distance_between_dists(v1,v2),k)

	dists_and_inds = sorted(dists_and_inds)
	return(dists_and_inds)


# Compute the p-values
def get_p_values_for_predictors(dataframe,predictors,target):

	y = dataframe[target]
	y = np.array(y).reshape(-1,1)

	X = dataframe[predictors]
	X = np.array(X).reshape(-1,len(predictors))

	# Use linear regression. Review the assumptions later. This may not be the right model.
	reg = sklearn.linear_model.LinearRegression()
	reg.fit(X,y)

	sse = np.sum((reg.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])

	se = np.array([
        np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.array(np.dot(X.T, X),dtype='float'))))
                                                for i in range(sse.shape[0])
                ])

	tt = reg.coef_ / se
	p = 2 * (1 - sp.t.cdf(np.abs(tt), y.shape[0] - X.shape[1]))

	return(reg,p)

# Get the significant predictors and compare how the quantiles present themselves
def get_significant_predictors(dataframe,predictors,target,significance_level=0.05):

	reg,p = get_p_values_for_predictors(dataframe,predictors,target)
	
	# Get the significant predictors
	significant_predictors_idx = [i for (i,pp) in enumerate(p[0]) if pp < significance_level]

	return([predictors[x] for x in significant_predictors_idx])

# Get a quick summary of the best and worst stats
def best_v_worst_stats(variables,group1,group2):
    dic = {}
    for v in variables:
        dic[v] = [np.mean(group1[v]),np.mean(group2[v])]
    
    return(dic)

# Determine the quantile of the patreon campaigns that you are in
def get_quantile(comparison_campaigns,target):
	
	if target.iloc[0]['earnings_visibility'] == 'private':
		return(-1)

	target_pledge_sum = target['pledge_sum']
	non_nulls = ~pd.isnull(comparison_campaigns['pledge_sum'])
	comparison_campaigns_pledges = comparison_campaigns[non_nulls]['pledge_sum'] 

	return(sum(comparison_campaigns_pledges < int(target_pledge_sum))/len(comparison_campaigns_pledges))


def get_all_recommendations(predictors, best, worst, target):

	recommendations = []
	bvw = best_v_worst_stats(predictors, best, worst)
	
	for key in bvw.keys():

		# If the best campaigns have more of something and the target campaign has less, then recommend more
		if(bvw[key][0] > bvw[key][1]) and (bvw[key][0] > int(target.iloc[0][key])):

			recommendations.append('{} ++'.format(key))

		# Else if the best campaigns have less of something and the target campaign has more, then recommend less
		elif(bvw[key][0] < bvw[key][1]) and (bvw[key][0] < int(target.iloc[0][key])):

			recommendations.append('{} --'.format(key))

	if len(recommendations) == 0:
		recommendations.append('We cannot offer any recommendations at this time.')

	return(recommendations)






















