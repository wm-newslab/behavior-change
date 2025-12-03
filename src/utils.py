import csv
import zlib #https://docs.python.org/3/library/zlib.html

from bloc.util import genericErrorInfo
from datetime import datetime
import matplotlib.font_manager as font_manager

from bloc.generator import add_bloc_sequences
from bloc.util import get_bloc_variant_tf_matrix
from bloc.util import get_bloc_variant_tf_matrix
from bloc.util import conv_tf_matrix_to_json_compliant
from bloc.util import cosine_sim
from bloc.util import genericErrorInfo


def parse_segments(segmented_bloc_string):
    """Split each field in the dictionary into a list of segments."""
    return {
        key: value.split('|')
        for key, value in segmented_bloc_string.items()
    }

def cal_cosine_sim(seg1, seg2):
    bloc_variant = {'type': 'folded_words', 'fold_start_count': 3, 'count_applies_to_all_char': False}
    bloc_params = {'ngram': 1, 'min_df': 1, 'tf_matrix_norm': '', 'keep_tf_matrix': True, 'set_top_ngrams': True, 'top_ngrams_add_all_docs': False, 'token_pattern': '[^□⚀⚁⚂⚃⚄⚅. |()*]+|[□⚀⚁⚂⚃⚄⚅.]'}

    cur_prev_tf_mat = get_bloc_variant_tf_matrix(
        [{'text': seg1}, {'text': seg2}], 
        ngram=bloc_params['ngram'], 
        min_df=bloc_params['min_df'], 
        tf_matrix_norm=bloc_params['tf_matrix_norm'], 
        keep_tf_matrix=bloc_params['keep_tf_matrix'],
        set_top_ngrams=bloc_params['set_top_ngrams'], 
        top_ngrams_add_all_docs=bloc_params['top_ngrams_add_all_docs'],
        bloc_variant=bloc_variant,
        token_pattern=bloc_params['token_pattern']
    )
    #  cur_prev_tf_mat => (tf_matrix, tf_matrix_normalized, tf_idf_matrix, vocab, top_ngrams, token_pattern)
    # print("tf_idf_matrix", cur_prev_tf_mat['tf_idf_matrix'][0]['tf_vector'])
    # print("tf_matrix", cur_prev_tf_mat['tf_matrix'][0]['tf_vector'])
    
    try:    
        matrix_form = f"tf_matrix"
        cur_prev_tf_mat = conv_tf_matrix_to_json_compliant(cur_prev_tf_mat)
        dist = 1 - cosine_sim( [cur_prev_tf_mat[matrix_form][0]['tf_vector']], [cur_prev_tf_mat[matrix_form][1]['tf_vector']] ) 
    except Exception as e:
        print("error", genericErrorInfo(e))
        dist = 0.0  
    return dist

def compressed_size(s):
    return len(zlib.compress(s.encode("utf-8")))

def cal_ncd(seg1, seg2):
    c1 = compressed_size(seg1)
    c2 = compressed_size(seg2)
    c12 = compressed_size(seg1 + seg2)

    raw_ncd = (c12 - min(c1, c2)) / max(c1, c2)
    
    return raw_ncd

def symmetric_ncd_matrix(lhs, rhs):
    d1 = cal_ncd(lhs, rhs)
    d2 = cal_ncd(rhs, lhs)
    change = (d1 + d2) / 2  # symmetrize
    return change

def calculate_change(lhs, rhs, method):
    if (not lhs or not rhs):
        return None
    if method == 'compression':
        dist = cal_ncd(lhs, rhs)
    else:
        dist = cal_cosine_sim(lhs, rhs)
    return dist

def adjacent_changes(segments, method):
    changes_list = []
    for i in range(len(segments) - 1):
        lhs = segments[i]
        rhs = segments[i + 1]
        dist = calculate_change(lhs, rhs, method)
        if dist is None:
            print(f"Skipping change calculation for empty segment pair: {lhs}, {rhs}")
            continue
        changes_list.append(dist)

    return changes_list

def pairwise_changes(segments, method):
    changes_list = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            lhs = segments[i]
            rhs = segments[j]
            dist = calculate_change(lhs, rhs, method)
            if dist is None:
                print(f"Skipping change calculation for empty segment pair: {lhs}, {rhs}")
                continue
            changes_list.append(dist)

    return changes_list 

def cumulative_previous_changes(segments, method):
    changes_list = []
    for i in range(1, len(segments)):
        lhs = ''.join(segments[0:i])
        rhs = segments[i]
        dist = calculate_change(lhs, rhs, method)
        if dist is None:
            print(f"Skipping change calculation for empty segment pair: {lhs}, {rhs}")
            continue
        changes_list.append(dist)

    return changes_list

def parse_time(t):
    return datetime.strptime(t, "%a %b %d %H:%M:%S %z %Y")

def get_user_id_class_map(f):
    user_id_class_map = {}
    all_classes = set()

    try:
        with open(f) as fd:
            
            rd = csv.reader(fd, delimiter='\t')
            for user_id, user_class in rd:
                user_id_class_map[user_id] = user_class
                all_classes.add(user_class)
    except:
        genericErrorInfo()

    return user_id_class_map, all_classes

def update_user_class(dataset_name, user_class):
    updated_user_class = user_class
    if dataset_name == 'astroturf':
        updated_user_class = 'bot'
    elif dataset_name == 'cresci-17' and (user_class == 'socialspam' or user_class == 'bot-traditionspam' or user_class == 'bot-fakefollower' or user_class == 'bot-socialspam'):
        updated_user_class = 'bot'
    elif dataset_name == 'zoher-organization' and user_class == 'organization':
        updated_user_class = 'human'
    return updated_user_class

def generate_bloc_for_all(records, gen_bloc_params, all_bloc_symbols):

    out = []
    for r in records:
        u_bloc = add_bloc_sequences(
            r["tweets"],
            all_bloc_symbols=all_bloc_symbols,
            **gen_bloc_params,
        )
        r2 = dict(r)
        r2["u_bloc"] = u_bloc
        out.append(r2)
    return out

def segment_bloc_for_all(records, segmentation_type, n_gram):
    out = []
    for r in records:
        u_bloc = r["u_bloc"]

        segmented = {
            "action": u_bloc["bloc"]["action"],
            "content_syntactic": u_bloc["bloc"]["content_syntactic"],
        }

        if segmentation_type == "sets_of_four":
            for alphabet in ["action", "content_syntactic"]:
                bloc_content = ""
                count = 0
                for tweet in u_bloc["tweets"]:
                    count += 1
                    seq = tweet["bloc"]["bloc_sequences_short"][alphabet]
                    if count >= n_gram:
                        bloc_content += seq + "|"
                        count = 0
                    else:
                        bloc_content += seq
                if bloc_content.endswith("|"):
                    bloc_content = bloc_content[:-1]
                segmented[alphabet] = bloc_content

        r2 = dict(r)
        r2["segmented_bloc_string"] = segmented
        out.append(r2)
    return out

def calculate_changes_for_all(records, comparison_method, change_calculation_method):
    out = []
    for r in records:
        seg = r["segmented_bloc_string"]
        parsed = parse_segments(seg)

        if comparison_method == "adjacent":
            action_changes = adjacent_changes(parsed["action"], change_calculation_method)
            content_changes = adjacent_changes(parsed["content_syntactic"], change_calculation_method)

        elif comparison_method == "pairwise":
            action_changes = pairwise_changes(parsed["action"], change_calculation_method)
            content_changes = pairwise_changes(parsed["content_syntactic"], change_calculation_method)

        elif comparison_method == "cumulative":
            action_changes = cumulative_previous_changes(parsed["action"], change_calculation_method)
            content_changes = cumulative_previous_changes(parsed["content_syntactic"], change_calculation_method)

        else:
            raise ValueError("Unknown comparison_method: " + comparison_method)

        r2 = dict(r)
        r2["action_changes_list"] = action_changes
        r2["content_changes_list"] = content_changes
        out.append(r2)
        
    return out
