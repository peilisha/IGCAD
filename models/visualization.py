import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors

def visualize_weights_all_leads(signals, attentions,idx):
    """
    Visualization of the signals of the 12 leads and their weight mapping
    Parameters
    ----------
    - signals:  (12, N) 12-lead signals
    - attentions:  (12, N) weights
    """
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_leads = 12
    window_size = 80
    step_size = 40
    plt.figure(figsize=(35, 24), constrained_layout=True)
    fig, axs = plt.subplots(num_leads, 1, figsize=(35, 24), gridspec_kw={'height_ratios': [1] * num_leads})
    for i in range(num_leads):
        signal = signals[i]
        attention = attentions[i]
        num_segments = (len(signal) - window_size) // step_size + 1
        weighted_signal = np.zeros_like(signal)
        weight_counts = np.zeros_like(signal)
        for j in range(num_segments):
            start = j * step_size
            end = start + window_size
            weighted_signal[start:end] += attention[j]
            weight_counts[start:end] += 1
        weight_counts[weight_counts == 0] = 1
        weighted_signal /= weight_counts
        axs[i].plot(signal, color='gray', alpha=0.3,linewidth=3)
        axs[i].scatter(np.arange(len(signal)), signal, c=weighted_signal, cmap='viridis', s=50)
        axs[i].set_title(f"Lead {leads[i]}")
    fig.subplots_adjust(hspace=0.4)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cbar_ax, orientation='horizontal')
    plt.savefig("all_leads_with_weights.svg", bbox_inches='tight', pad_inches=0)




