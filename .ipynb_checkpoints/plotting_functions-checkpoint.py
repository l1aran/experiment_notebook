import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_df(data, labels, use_colors=True):
    """
    Plot data from a DataFrame where the first row contains labels and the remaining rows contain data.

    Parameters:
        data (pd.DataFrame or np.ndarray): DataFrame or array with sine waves as rows.
        labels (np.ndarray): Array of labels for each sine wave.
        use_colors (bool): If True, different colors for each cluster. If False, all sine waves will be the same color.
    """
    # Ensure labels are integers
    labels = labels.astype(int)
    
    # Transpose data if it is a DataFrame to ensure each sine wave is a row
    if isinstance(data, pd.DataFrame):
        data = data.values.T

    unique_labels = np.unique(labels)
    if use_colors:
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        label_color_map = dict(zip(unique_labels, colors))
    else:
        label_color_map = {label: 'navy' for label in unique_labels}

    plt.figure(figsize=(15, 5))
    for i in range(data.shape[0]):
        plt.plot(data[i], color=label_color_map[labels[i]], linestyle='-', marker='o')

    if use_colors:
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=label_color_map[label], lw=2) for label in unique_labels]
        plt.legend(custom_lines, [f'Label {label}' for label in unique_labels])

    plt.title('Generated Sine Waves')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.show()

def plot_cluster_variables(simulations, cluster_variables, ax, states_from_chain=None):
    'view latent state of an entire cluster'
    # Define a colormap that provides a distinct color for each variable.
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_variables)))  # Use the 'viridis' colormap for distinct colors.

    for idx, variable in enumerate(cluster_variables):
        color = colors[idx]
        # if variable in [0, 1]:
        #     color = 'red'
        # else:
        #     color = colors[idx]
        plot_predictions_with_labels(simulations, variable, ax, color, states_from_chain)

    ax.set_title("Clustered Variables Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.legend(loc='upper left', handletextpad=0, frameon=True, fontsize='small')

def compare_clusters(data, labels, cluster, idx1, idx2):
    """
    Plots data for specified clusters with special highlighting for two sets of indices.
    
    Parameters:
        data (list or array): The dataset to plot.
        labels (list or array): The labels for the data points.
        cluster (list): List of cluster indices to highlight.
        idx1 (list): Indices of data points to highlight in the first plot.
        idx2 (list): Indices of data points to highlight in the second plot.
    """
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))
    
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))  # 1 row, 2 columns
    
    # Plot for idx1 highlighted data points
    for i in range(len(labels)):
        if labels[i] in cluster:
            if i in idx1:
                ax[0].plot(data[i], color='blue', linestyle='-', marker='o')
            else:
                ax[0].plot(data[i], color=label_color_map[labels[i]], linestyle='-', marker='o')    

    # Plot for idx2 highlighted data points
    for i in range(len(labels)):
        if labels[i] in cluster:
            if i in idx2:
                ax[1].plot(data[i], color='blue', linestyle='-', marker='o')
            else:
                ax[1].plot(data[i], color=label_color_map[labels[i]], linestyle='-', marker='o')
    
    plt.show()

def plot_latent_state_sequence(timesteps, values, states, ax):
    assert len(timesteps) == len(states)
    unique = sorted(set(states))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique)))
    y_low, y_high = ax.get_ylim()
    y_mid = np.mean([y_low, y_high])
    y_height = 0.05 * (y_high - y_low)
    for state, color in zip(unique, colors):
        xs = timesteps[states == state]
        for x in xs:
            ax.fill_between([x-1, x], [y_mid-y_height]*2, [y_mid+y_height]*2,
                            alpha=0.3, color=color)

def plot_predictions(model, variable, ax, states_from_chain=None):
    index = model.variables.index(variable)
    # Plot the observed data.
    x_observed = model.dataset.index
    y_observed = model.dataset.loc[:, variable]
    ax.plot(x_observed, y_observed, label=variable, color='k', linewidth=1)
    ax.set_ylim([min(y_observed)-2, max(y_observed)+2])
    
    # Optionally plot latent temporal state at each timepoint,
    # according to a given chain in the model.
    if states_from_chain is not None:
        assert 0 <= states_from_chain < model.chains
        states = model.get_temporal_regimes(variable)[states_from_chain]
        plot_latent_state_sequence(x_observed, y_observed, states, ax)
    
    # Add the legend.
    ax.legend(loc='upper left', handletextpad=0)

def plot_predictions_with_labels(model, variable, ax, color, states_from_chain=None):
    index = model.variables.index(variable)
    x_observed = model.dataset.index
    y_observed = model.dataset.loc[:, variable]

    # Plot observed data with a label at the end of the line.
    line, = ax.plot(x_observed, y_observed, label=variable, color=color, linewidth=1.5)
    label_x_pos = x_observed[-1]
    label_y_pos = y_observed.iloc[-1]

    # Optional: plot the latent temporal state at each timepoint.
    if states_from_chain is not None:
        assert 0 <= states_from_chain < model.chains
        states = model.get_temporal_regimes(variable)[states_from_chain]
        plot_latent_state_sequence(x_observed, y_observed, states, ax)
