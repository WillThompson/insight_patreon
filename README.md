# *Patreon Pro*: Recommendations for content creators!

In this repository, you will find the underlying code for my project, *Patreon Pro*. It is an application created for content creators fundraising on Patreon.com to compare the content of their patreon page to other working in similar content spaces (i.e. podcasting, visual arts, even software!). Essentially, the app will scan and analyze the summary of a creator's campaign to characterize its content. It will then compare to a large sample of other patreon campaigns, find those that are 'most similar' in terms of content, determine which factors are significant in terms of soliciting funding, and display recommendations to the user.

The live version of the project is located at: [http://www.patreonpro.site:5000](http://www.patreonpro.site:5000)

## Rationale: Why did I want to make this?

I was interested in exploring natural language processing and topic modelling and I decided that a good way to do this was by looking at crowdfunding campaigns.

As far as I can tell, Patreon does not offer recommendations for content creators on its platform that are specific to the type of content that the users are creating, although they do point to generally successful campaigns (If I am wrong, please correct me). Content creators can search by keywords in the Patreon search bar to find similar content, but getting a comprehensive summary of similar campaigns has not been implemented.

## Getting the data

Having scraping the (legacy) Patreon API for campaign and curator data, I retrieved several hundred thousand records. 

Each record is associated with a fundraising campaign on Patreon.com. The related query returns a JSON response with campaign information, but also data on the curator and the goals of the campaign and rewards for contributing (if any).

## How does Patreon Pro work?

In the data, campaigns have a field where the curator of a campaign can offer a summary. This summary is the most descriptive element of a campaign and it is on this basis that we can find similar campaigns. Using *Term-Frequency-Inverse-Document-Frequency* (*TF-IDF*) vectorization for summaries, we can determine which words appear most often across different summaries, appropriately scaled by the number of documents they appear in. This vectorization of the documents is then used to train a *Latent Dirichlet Allocation* (*LDA*) model to find 'latent topics' in the summaries. This LDA model is used to project the summary of a Patreon campaign onto a 'topic space' in which we have some notion of 'distance' between campaigns.

Thus, we can pass a new campaign into our TF-IDF vectorization scheme and trained LDA model and determine which campaigns in our dataset are closest to the new one. We choose to use is the *Jensen-Shannon distance* (basically a symmetrized version of the *Kullback–Leibler distance*) to define the distance between the topic distributions. 

Once we have determined the closest campaigns, we perform a quick regression analysis on the group to determine which factors are statistically significant for increased funding.

## Improvements

+ Additional predictors for fundraising success: There are additional features that can be enginnered into the model.
+ Better validation of the topic modelling: Topic modelling is an unsupervised learning technique. As such, there are fewer ways to validate that the technique has been successful. Currently, I have been checking consistency of generated topics across different half-partitions of the dataset to ensure that similar topics are generated consistently. Generally, this has been successful. (Results to be described).


