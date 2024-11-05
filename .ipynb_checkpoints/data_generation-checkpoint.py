import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_sine_wave(num_points, frequency, amplitude, phase, noise_level, rng):
    """
    Generate a sine wave with specified parameters and noise level.

    Parameters:
        num_points (int): Number of data points.
        frequency (float): Frequency of the sine wave.
        amplitude (float): Amplitude of the sine wave.
        phase (float): Phase shift of the sine wave.
        noise_level (float): Standard deviation of Gaussian noise added to the sine wave.
        rng (np.random.Generator): Random number generator for reproducibility.

    Returns:
        np.ndarray: Generated sine wave with noise.
    """
    time = np.linspace(0, 1, num_points)
    noise = rng.normal(0, noise_level, num_points)
    return amplitude * np.sin(2 * np.pi * frequency * time + phase) + noise

def generate_sine_wave_data(num_samples=50, num_points=100, num_clusters=5, noise_level=None, plot=True,
                            frequency_noise=0.1, amplitude_noise=0.1, phase_noise=0.1, random_state=None,
                            use_colors=True, fixed_amplitude = None):
    """
    Generate a dataset of sine waves with specified parameters and plot the waves.

    Parameters:
        num_samples (int): Number of sine wave samples.
        num_points (int): Number of data points per sine wave.
        num_clusters (int): Number of clusters (distinct sets of sine wave parameters).
        noise_level (float): Standard deviation of Gaussian noise added to each sine wave.
        plot (bool): Whether to plot the generated sine waves.
        frequency_noise (float): Noise level for frequency.
        amplitude_noise (float): Noise level for amplitude.
        phase_noise (float): Noise level for phase.
        random_state (int or None): Seed for the random number generator for reproducibility.
        colors (bool): If True, different colors for each cluster. If False, all sine waves will be the same color.

    Returns:
        pd.DataFrame: DataFrame with sine waves as columns, index as integers, and first row as labels.
    """
    rng = np.random.default_rng(random_state)

    if noise_level is None:
        noise_level = rng.uniform(0.1, 0.5)

    labels = np.arange(num_samples) % num_clusters

    data = []
    frequencies = np.linspace(1, 10, num_clusters)
    amplitudes = np.linspace(1, 5, num_clusters) if fixed_amplitude is None else np.full(num_clusters, fixed_amplitude)
    phases = np.linspace(0, 2 * np.pi, num_clusters)

    for i in range(num_samples):
        cluster_idx = labels[i]

        # Add variation
        frequency = frequencies[cluster_idx] + rng.normal(0, frequency_noise)
        amplitude = amplitudes[cluster_idx] + rng.normal(0, amplitude_noise)
        phase = phases[cluster_idx] + rng.normal(0, phase_noise)
    
        sine_wave = generate_sine_wave(num_points, frequency, amplitude, phase, noise_level, rng)
        data.append(sine_wave)

    # Convert data to DataFrame
    df = pd.DataFrame(data).T
    df.columns = [i for i in range(num_samples)]
    
    # Add labels as a new row at the top
    df.loc[-1] = labels
    df.index = df.index + 1  # Shift index to make space for labels
    df = df.sort_index()

    # Convert labels row to integers
    df.iloc[0] = df.iloc[0].astype(int)

    if plot:
        if use_colors:
            unique_labels = np.unique(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
            label_color_map = dict(zip(unique_labels, colors))
        else:
            label_color_map = {label: 'navy' for label in np.unique(labels)}
        
        plt.figure(figsize=(15, 5))
        for i in range(num_samples):
            plt.plot(df[i][1:], color=label_color_map[labels[i]], linestyle='-', marker='o')
        
        if use_colors:
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color=label_color_map[label], lw=2) for label in unique_labels]
            plt.legend(custom_lines, [f'Label {label}' for label in unique_labels])
        
        plt.title('Generated Sine Waves')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')

    return df

def generate_data_3(n_series=5, n_observations=100, seed=42):
    np.random.seed(seed)
    data = {}

    for i in range(n_series):
        if i == n_series - 1:
            # Outlier with significantly different signal
            trend = np.linspace(0, 50, n_observations)
            seasonal = 20 * np.sin(np.linspace(0, 6 * np.pi, n_observations))
            noise = np.random.normal(0, 5, n_observations)
        elif i in [n_series - 2, n_series - 3]:
            # Cluster 2 with distinct signal
            trend = np.linspace(0, 30, n_observations)
            seasonal = 10 * np.sin(np.linspace(0, 4 * np.pi, n_observations))
            noise = np.random.normal(0, 3, n_observations)
        else:
            # Cluster 1 with standard signal
            trend = np.linspace(0, 10, n_observations)
            seasonal = 5 * np.sin(np.linspace(0, 3 * np.pi, n_observations))
            noise = np.random.normal(0, 1, n_observations)

        data[f'Series_{i + 1}'] = trend + seasonal + noise

    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    df = generate_sine_wave_data(random_state=42)
    print(df)
