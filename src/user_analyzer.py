import gzip
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from bloc.util import get_default_symbols
from .utils import calculate_changes_for_all, segment_bloc_for_all, generate_bloc_for_all

def plot_histogram(ax, data, alphabet, bin_edges):
    color = '#2723eb'
    hatch_color = 'white'

    # use provided bin_edges if given, else compute locally
    if bin_edges is None:
        bin_edges = np.linspace(min(0, np.min(data)), max(1, np.max(data)), num=20)

    n, bins, patches = ax.hist(
        data,
        bins=bin_edges,
        color=color,
        linewidth=0.3,
        density=True,
        edgecolor='black',
    )

    # hatch pattern for content
    if alphabet == "content":
        for p in patches:
            p.set_hatch('///')
            p.set_facecolor(color)
            p.set_edgecolor(hatch_color)
            p.set_linewidth(0.5)

    # formatting that applies to all subplots
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    ax.set_title(
        f"{alphabet} alphabet", 
        fontsize=28,
    )

def plot_changes_distribution(changes_list):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    all_actions = changes_list['action_changes_list']
    all_contents = changes_list['content_changes_list']

    action_bins = np.linspace(
        min(0, np.min(all_actions)),
        max(1, np.max(all_actions)),
        num=10
    )
    content_bins = np.linspace(
        min(0, np.min(all_contents)),
        max(1, np.max(all_contents)),
        num=10
    )

    fig, axes = plt.subplots(
        1, 2, 
        figsize=(6*2, 4*1), 
        squeeze=False, 
        sharey=False,
        gridspec_kw={'hspace': 0.4}
    )

    for alphabet, index in [('action', 0), ('content', 1)]:
        if alphabet == 'action':
            change_values = all_actions
            bins = action_bins
        else:
            change_values = all_contents
            bins = content_bins

        ax = axes[0][index]

        plot_histogram(
            ax,
            change_values,
            alphabet=alphabet,
            bin_edges=bins
        )
        
        ax.set_xlabel("Distance", fontsize=28)
        ax.set_ylabel("Density", fontsize=28)

    # layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.4)
    plt.savefig("results/changes_distribution.png", dpi=300)

def read_user_data(config):
    path_to_dataset = config.get("path_to_data")

    tweets = []
    with gzip.open(path_to_dataset, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tweets.append(data)
    records  = [{
        "user_id": 'unknown',
        "user_class": 'unknown',
        "src": "unknown",
        "tweets": tweets,
    }]

    return records

def main(cfg):
    records = read_user_data(cfg)

    segmentation_type = cfg.get("segmentation_type")
    comparison_method = cfg.get("comparison_method")
    distance_metric = cfg.get("distance_metric")
    n_gram = cfg.get("n_gram")
    filename_for_bloc_params = f"gen_bloc_params_{'segment_on_pauses' if segmentation_type == 'sets_of_four' else segmentation_type}"
    gen_bloc_params = cfg.get(filename_for_bloc_params, {})
    all_bloc_symbols = get_default_symbols()

    records = generate_bloc_for_all(records, gen_bloc_params, all_bloc_symbols)
    records = segment_bloc_for_all(records, segmentation_type, n_gram)
    records = calculate_changes_for_all(records, comparison_method, distance_metric)

    plot_changes_distribution(records[0])
