import gzip
import os
import pandas as pd

from bloc.generator import add_bloc_sequences 
from bloc.util import get_default_symbols, getDictFromJson, genericErrorInfo

from .utils import get_user_id_class_map, update_user_class
from .utils import parse_time
from .utils import calculate_changes_for_all, segment_bloc_for_all, generate_bloc_for_all
from .classifier import classifier


def read_all_users(config, min_tweets_per_user, max_tweets_per_user):
    src = config.get("src")
    path_to_dataset = config.get("retraining_dataset")

    records = []

    for tweets_file in src:
        print(f"processing {tweets_file} accounts")
        file_name = f"{path_to_dataset}/{tweets_file}/tweets.jsons.gz"
        classname_file = f"{path_to_dataset}/{tweets_file}/userIds.txt"

        user_id_class_map, all_classes = get_user_id_class_map(classname_file)

        encoding = None
        if "stock" in tweets_file:
            encoding = "windows-1252"

        with gzip.open(file_name, "rt", encoding=encoding) as infile:
            for line in infile:
                try:
                    parts = line.split("\t")
                    if len(parts) != 2:
                        continue

                    user_id = parts[0]
                    tweets = getDictFromJson(parts[1])

                    if len(tweets) < min_tweets_per_user:
                        continue

                    user_class = user_id_class_map[user_id]
                    user_class = update_user_class(tweets_file, user_class)

                    tweets_sorted = sorted(tweets, key=lambda x: parse_time(x["created_at"]))
                    latest_tweets = tweets_sorted[-max_tweets_per_user:]

                    records.append(
                        {
                            "user_id": user_id,
                            "user_class": user_class,
                            "src": tweets_file,
                            "tweets": latest_tweets,
                        }
                    )
                except:
                    genericErrorInfo()

    return records


def main(cfg):
    segmentation_type = cfg.get("segmentation_type")
    comparison_method = cfg.get("comparison_method")
    distance_metric = cfg.get("distance_metric")
    n_gram = cfg.get("n_gram")
    filename_for_bloc_params = f"gen_bloc_params_{'segment_on_pauses' if segmentation_type == 'sets_of_four' else segmentation_type}"
    gen_bloc_params = cfg.get(filename_for_bloc_params, {})
    min_tweets_per_user = cfg.get("min_tweets_per_user", 20)
    max_tweets_per_user = cfg.get("max_tweets_per_user", 300)
    all_bloc_symbols = get_default_symbols()

    records = read_all_users(cfg, min_tweets_per_user, max_tweets_per_user)
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
    print(df[['user_id', 'user_class', 'src']].to_csv("dataset/retraining_user_id.txt", index=False))
    
    classifier(df, "automation_detection")
