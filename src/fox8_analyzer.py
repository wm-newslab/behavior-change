import gzip
import pandas as pd
import json
import os

from bloc.generator import add_bloc_sequences 
from bloc.util import get_default_symbols, getDictFromJson, genericErrorInfo

from .utils import get_user_id_class_map, update_user_class
from .utils import parse_time
from .utils import calculate_changes_for_all, segment_bloc_for_all, generate_bloc_for_all
from .classifier import classifier


def read_all_users(config):
    path_to_dataset = config.get("fox8_dataset")

    records = []

    with gzip.open(path_to_dataset, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())  

                records.append(
                    {
                        "user_id": data['user_id'],
                        "user_class": 'bot' if (data['dataset'] == "fox8") else 'human',
                        "src": data['dataset'],
                        "tweets": data['user_tweets'],
                    }
                )
            except:
                genericErrorInfo()

    return records


def main(cfg):
    records = read_all_users(cfg)

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

    # build final user_data for classifier
    user_data = []
    for r in records:
        seg = r["segmented_bloc_string"]
        if(len(r["action_changes_list"]) >= 2 or len(r["content_changes_list"])  >= 2):
            user_data.append(
                {
                    "user_class": r["user_class"],
                    "src": r["src"],
                    "action_changes_list": r["action_changes_list"],
                    "content_changes_list": r["content_changes_list"],
                    "action_bloc": seg["action"],
                    "content_bloc": seg["content_syntactic"],
                    "user_id": r["user_id"],
                }
            )

    df = pd.DataFrame(user_data)

    #used for save user_id set
    print(df[['user_id', 'user_class']].to_csv("dataset/fox8_user_id.txt", index=False))
    
    classifier(df, "coordination_detection")
