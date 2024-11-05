import os
import sys

# Add the parent directory to sys.path
sys.path.append('/hdsi-collab-persistent/cephfs/MVP/trcrpm')


import numpy as np
import time
from sklearn.metrics import adjusted_rand_score
from src import Hierarchical_TRCRP_Mixture, TRCRP_Mixture
from IPython.utils.capture import capture_output


def run_model(data, num_chains = 8, p = 5, MCMC_steps = 1000, hyperparam_steps = 50, runtime = True):
    '''
    incorporates observations into the model, cycles through inference kernels, and resamples latent parameters
    
    parameters
        data: df, data.columns are assumed to be time series
        chains: number of Markov Chains used for inference
        lag: number of time points to use in reweighting CRP. if lag=0 -> no temporal aspect
        MCMC_steps: number of steps for MCMC inference
        hyperparam_steps: number of steps for hyperparameter optimization

    returns:
        TRCRP model
    
    '''
    
    
    # fit the model
    model = Hierarchical_TRCRP_Mixture(chains = num_chains, variables = data.columns.tolist(), 
                                       rng = np.random.RandomState(42), lag = p)
    # lag = 0 -> standard CRP mixture
    model.incorporate(data)

    with capture_output() as captured:
        start = time.time()
        model.resample_all(steps = MCMC_steps) # burn-in period
        end = time.time()
    if runtime:
        elapsed_time = end - start
        minutes = int(elapsed_time // 60)  # Get the full minutes
        seconds = int(elapsed_time % 60)   # Get the remaining seconds
        print(f'MCMC inference time: {minutes} minutes and {seconds} seconds')
    
    with capture_output() as captured:
        start = time.time()
        model.resample_hyperparameters(steps = hyperparam_steps)
        end = time.time()
    if runtime:
        elapsed_time = end - start
        minutes = int(elapsed_time // 60)  # Get the full minutes
        seconds = int(elapsed_time % 60)   # Get the remaining seconds
        print(f'Hyperparameter optimization time: {minutes} minutes and {seconds} seconds')

    # probes = model.dataset.index
    # samples = model.simulate(probes, model.variables, numsamples)
    return model


def post_dep(model, num_samples, runtime = True):
    '''
    generates 'num_samples' posterior dependence probabilities between time series
    to obtain cluster assignments, we average dependence probabilites over posterior samples
     
    parameters
        model: TRCRP model
        num_samples: number of posterior samples

    returns
        if output = False:
            returns 3D array containing pairwise dependence probabilities of time series in each chain.
        the dimensions are (num_chains, len(data), len(data))
            result[i, j, k] ==1 if df[j] and df[k] are dependent (clustered) in Markov Chain[i]

    '''
    
    post_dep = [] # stores 'num_samples' posterior dependence matrices
    
    with capture_output() as captured: # supress output
        start = time.time()
        for _ in range(num_samples):
            model.resample_all(steps = 1)
            post_dep.append(model.dependence_probability_pairwise())
        end = time.time()

    if runtime:
        elapsed_time = end - start
        minutes = int(elapsed_time // 60)  # Get the full minutes
        seconds = int(elapsed_time % 60)   # Get the remaining seconds
        print(f'Sampling Time: {minutes} minutes and {seconds} seconds')


    return post_dep


def clustering(post_probs, threshold = 0.75):
    '''
    average dependence probabilites over the posterior samples and return clusters
    
    parameters:
        - post_probs: 3D array containing pairwise dependence probabilities of time series in each chain.
        the dimensions are (num_chains, len(data), len(data))
        - threshold: threshold for dependence; if data[j] and data[k] are dependent in threshold% of samples, 
            they are clustered together

    returns:
        clusters: list of indices that belong to each cluster
    '''
    post_probs = np.array(post_probs)
    avg_dep = np.mean(post_probs[:, :, :], axis = (0, 1)) # average across all markov chains for each pair

    import networkx as nx
    G = nx.Graph()
    n_var = len(avg_dep)
    variables = range(n_var) # can maybe modify this to contain column names

    for i in range(n_var):
        for j in range(i+1, n_var): # avoid self-loops and ensure i!=j
            if avg_dep[i, j] >= threshold:
                G.add_edge(variables[i], variables[j])


    for i in range(n_var):
        if i not in G:
            G.add_node(i)

    clusters = list(nx.connected_components(G))

    #print(f'Predicted Clusters w {int(threshold*100)}% threshold', threshold)
    # for i, cluster in enumerate(clusters):
    #     print(f"Cluster {i}: {sorted(cluster)}")


    return clusters

def return_ari(true_labels, predicted_clusters):
    '''
    parameters:
    - true_labels: column of true label values len(true_labels)==len(model.variables)
    - predicted_clusters: predicted clusters from clustering() functions

    returns:
    - adjusted randomized index score
    '''
    predicted_labels = [-1]*len(true_labels)
    for cluster_id, cluster in enumerate(predicted_clusters):
        for index in cluster:
            predicted_labels[index] = cluster_id

    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari