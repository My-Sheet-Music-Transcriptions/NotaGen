gt_feature_folder = "../clamp2/feature/schubert_interleaved"
output_feature_folder = "../clamp2/feature/weights_notagen_schubert-RL2_beta_0.1_lambda_10_p_size_16_p_length_1024_p_layers_20_h_size_1280_lr_1e-06_k_9_p_0.9_temp_1.2"
output_original_abc_folder = "../output/original/weights_notagen_schubert-RL2_beta_0.1_lambda_10_p_size_16_p_length_1024_p_layers_20_h_size_1280_lr_1e-06_k_9_p_0.9_temp_1.2"
output_interleaved_abc_folder = "../output/interleaved/weights_notagen_schubert-RL2_beta_0.1_lambda_10_p_size_16_p_length_1024_p_layers_20_h_size_1280_lr_1e-06_k_9_p_0.9_temp_1.2"
data_index_path = "schubert_RL3.json"
data_select_portion = 0.1

import os
import re
import json
import numpy as np
from config import *
from abctoolkit.check import check_alignment_unrotated
from abctoolkit.rotate import unrotate_abc


def load_npy_files(folder_path_list):
    """
    Load all .npy files from a specified folder and return a list of numpy arrays.
    """
    npy_list = []
    for file_path in folder_path_list:
        if file_path.endswith(".npy"):
            # file_path = os.path.join(folder_path, file_name)
            np_array = np.load(file_path)[0]
            npy_list.append(np_array)
    return npy_list


def average_npy(npy_list):
    """
    Compute the average of a list of numpy arrays.
    """
    return np.mean(npy_list, axis=0)


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two numpy arrays.
    """
    dot_product = np.dot(vec1, vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    cosine_sim = dot_product / (norm_vec1 * norm_vec2)

    return cosine_sim


def generate_preference_dict():
    gt_feature_paths = []
    for gt_feature_file in os.listdir(gt_feature_folder):
        gt_feature_paths.append(os.path.join(gt_feature_folder, gt_feature_file))
    gt_features = load_npy_files(gt_feature_paths)
    gt_avg_feature = average_npy(gt_features)

    output_feature_sim_dict = {}
    for file in os.listdir(output_feature_folder):
        output_feature_path = os.path.join(output_feature_folder, file)
        output_feature = np.load(output_feature_path)[0]
        sim = cosine_similarity(gt_avg_feature, output_feature)
        output_feature_sim_dict[file[:-4]] = sim

    threshold = int(len(output_feature_sim_dict) * data_select_portion)
    sorted_output_files = sorted(
        output_feature_sim_dict.keys(),
        key=lambda item: output_feature_sim_dict[item],
        reverse=True,
    )

    chosen_index = 0
    i = 0
    chosen_abc_paths = []
    while chosen_index < threshold and i < len(sorted_output_files):
        chosen_flag = True

        file = sorted_output_files[i]
        output_interleaved_abc_path = os.path.join(
            output_interleaved_abc_folder, file + ".abc"
        )

        with open(output_interleaved_abc_path, "r") as f:
            abc_lines = f.readlines()

        # check aligment
        try:
            abc_lines_unrotated = unrotate_abc(abc_lines)
            barline_equal_flag, bar_no_equal_flag, bar_dur_equal_flag = (
                check_alignment_unrotated(abc_lines_unrotated)
            )
            if not (barline_equal_flag and bar_no_equal_flag and bar_dur_equal_flag):
                raise Exception
        except:
            chosen_flag = False

        # check header: sheets where staves for the same instrument are not grouped together are excluded from the chosen set.
        appeared_inst = set()
        last_inst = ""
        for line in abc_lines:
            if line.startswith("V:") and "nm=" in line:
                match = re.search(r'nm="([^"]+)"', line)
                if match:
                    inst = match.group(1)
                    if inst != last_inst and inst in appeared_inst:
                        chosen_flag = False
                        break
                    else:
                        last_inst = inst
                        appeared_inst.add(inst)

        # check plagiarism: sheets with sim > 0.95 are excluded
        output_feature_path = os.path.join(output_feature_folder, file + ".npy")
        output_feature = np.load(output_feature_path)[0]
        for gt_feature_file in os.listdir(gt_feature_folder):
            gt_feature_path = os.path.join(gt_feature_folder, gt_feature_file)
            gt_feature = np.load(gt_feature_path)[0]
            sim = cosine_similarity(output_feature, gt_feature)
            if sim > 0.95:
                chosen_flag = False
                break

        if chosen_flag:
            original_abc_path = os.path.join(output_original_abc_folder, file + ".abc")
            chosen_abc_paths.append(original_abc_path)
            chosen_index += 1
        else:
            print(file, "skipped")

        i += 1

    rejected_abc_paths = [
        os.path.join(output_original_abc_folder, file + ".abc")
        for file in sorted_output_files[-threshold:]
    ]
    preference_dict = {"chosen": chosen_abc_paths, "rejected": rejected_abc_paths}

    with open(data_index_path, "w") as w:
        json.dump(preference_dict, w, indent=4)


if __name__ == "__main__":
    generate_preference_dict()
