import os, json, argparse
import pandas as pd
import numpy as np
from datetime import datetime
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "--config",
        type=str,
        help="config",
        required=True,
    )
    return parser.parse_args()


def main():
    print(">> Start comprehension")
    args = parse_args()

    with open(args.config, "r") as f:
        path_config = json.load(f)

    PATH = path_config["PATH"]
    LABEL = path_config["LABEL"]
    path_data = os.path.join(PATH, "data")
    path_result = os.path.join(PATH, "clesh_results")
    path_save = PATH

    analysis_config = path_config.get("ANALYSIS", {})
    cont_bound = analysis_config.get("cont_bound", 10)  # Default=10
    candidate_num_min = analysis_config.get("candidate_num_min", 10)  # Default=10
    candidate_num_max = analysis_config.get("candidate_num_max", 20)  # Default=20
    p_feature_selection = analysis_config.get(
        "p_feature_selection", 0.05
    )  # Default=0.05
    manual_num = analysis_config.get("manual_num", 0)  # Default=0
    interaction_list = analysis_config.get("interaction_list", False)  # Default=False
    p_univariate = analysis_config.get("p_univariate", 0.05)  # Default=0.05
    p_interaction = analysis_config.get("p_interaction", 0.05)  # Default=0.05

    # Load data
    data_feat = pd.read_csv(os.path.join(path_data, "features.csv"))
    data_feat = data_feat.drop(data_feat.columns[0], axis=1)
    data_shap = np.load(os.path.join(path_data, "shap.npy"))

    feat_type = np.load(os.path.join(path_result, "_tmp", "feat_type.npy"))
    rank_feat = np.load(os.path.join(path_result, "_tmp", "rank_feat.npy"))
    idx_pair = np.load(os.path.join(path_result, "_tmp", "idx_pair.npy"))

    type_dict = {0: "discrete", 1: "continuous", 2: "binary"}
    avg_class = {0: "upper", 1: "lower"}

    report = open(os.path.join(path_save, "clesh_report.md"), "w")

    report.write(f"# CLE-SH results for {LABEL}\n")
    report.write(f'Date of the report: {datetime.today().strftime("%Y.%m.%d")}\n\n')

    report.write(f"## Characteristic of Dataset\n")
    report.write(
        f"Number of samples: {len(data_feat.index)}, Number of features: {len(data_feat.columns)}\n\n"
    )

    report.write(f"## Feature Selection\n")
    report.write(
        f"Among entire features, {len(rank_feat)} features were selected as important features, showing higher importance than other features (p < {p_feature_selection})\n\n"
    )

    n_type = [0, 0, 0]
    for idx in rank_feat:
        n_type[feat_type[idx]] += 1

    report.write(
        f"(Binary: {n_type[2]}, Discrete: {n_type[0]}, continuous: {n_type[1]})\n\n"
    )

    report.write(f"### SHAP Summary Plot\n")
    report.write('<p align="center">\n')
    report.write(f'<img src="./clesh_results/shap_summary_plot.jpg" width="500">\n')
    report.write("</p>\n\n")

    report.write(f"## Univariate Analysis\n")
    files = os.listdir(os.path.join(path_result, "univariate_analysis"))

    check_tukey = []
    for item in files:
        if item[:5] == "tukey":
            check_tukey.append(item[6:-4])

    for i_rank, i_t in enumerate(rank_feat):
        name_target = data_feat.columns[i_t]

        report.write(
            f"### rank {i_rank+1}. {name_target}({type_dict[feat_type[i_t]]})\n"
        )
        report.write('<p align="center">\n')
        report.write(
            f'<img src="./clesh_results/univariate_analysis/{name_target}.jpg" height="300">\n'
        )
        report.write("</p>\n\n")

        if name_target in check_tukey:
            report.write('<p align="center">\n')
            report.write(
                f'<img src="./clesh_results/univariate_analysis/tukey_{name_target}.jpg" height="300">\n'
            )
            report.write("</p>\n\n")

        read = open(path_result + f"/univariate_analysis/_tmp/{name_target}.txt", "r")
        list_line = []

        while True:
            line = read.readline()
            if not line:
                break
            list_line.append(line)
        read.close()

        if feat_type[i_t] != 1:
            list_decrease = []
            list_increase = []
            for line in list_line:
                if line[:8] == "decrease":
                    list_decrease.append(line[10:-1])
                elif line[:8] == "increase":
                    list_increase.append(line[10:-1])

            report.write(
                f'but category {", ".join(list_increase)} **increases** the possibility (p < {p_univariate})\n\n'
            )

            line_split = list_line[-1].split(" ")

            if line_split[0] == "discrete":
                if line_split[2][:-1] == "True":
                    report.write(
                        f"(Between categories) There are statistical significant differences between the importance of categories as following (p < {p_univariate})\n"
                    )
                else:
                    report.write(
                        f"(Between categories) There are **no** statistical significant differences between the importance of categories as following (p > {p_univariate})\n"
                    )

            elif line_split[0] == "binary":
                if line_split[2][:-1] == "True":
                    report.write(
                        f"(Between categories) There are statistical significant differences between the importance of binary categories (p < {p_univariate})\n"
                    )
                else:
                    report.write(
                        f"(Between categories) There are **no** statistical significant differences between the importance of binary categories (p < {p_univariate})\n"
                    )

        else:
            if list_line[0].split(" ")[1] != "None":
                report.write(
                    f'SHAP values follow the {list_line[0].split(" ")[1]} function with the statistical significance (p < {p_univariate})\n'
                )
            else:
                report.write(
                    f"SHAP values do not follow any function with the statistical significance (p > {p_univariate})\n"
                )

    report.write(f"## Interactive Analysis\n")
    files = os.listdir(f"{path_result}/interactive_analysis")

    check_tukey = []
    dict_tukey = {}
    for item in files:
        item_split = item.split(sep="_")
        if item_split[0] == "tukey":
            check_tukey.append(item_split[1])
            if item_split[1] not in dict_tukey:
                dict_tukey[item_split[1]] = [item]
            else:
                dict_tukey[item_split[1]].append(item)

    for i_rank, (i_t, i_i) in enumerate(idx_pair):
        name_target = data_feat.columns[i_t]
        name_interaction = data_feat.columns[i_i]

        report.write(
            f"### rank {i_rank+1}. {name_target}({type_dict[feat_type[i_t]]}), Interaction: {name_interaction}({type_dict[feat_type[i_i]]})\n"
        )
        target_value_avg = np.average(data_feat.iloc[:, i_i])

        report.write('<p align="center">\n')
        report.write(
            f'<img src="./clesh_results/interactive_analysis/{name_target}_{name_interaction}.jpg"; height="300">\n'
        )
        report.write("</p>\n\n")

        if name_target in check_tukey:
            for item in dict_tukey[name_target]:
                report.write('<p align="center">\n')
                report.write(
                    f'<img src="./clesh_results/interactive_analysis/{item}" height="300">\n'
                )
                report.write("</p>\n\n")

        read = open(path_result + f"/interactive_analysis/_tmp/{name_target}.txt", "r")
        list_line = []

        while True:
            line = read.readline()
            if not line:
                break
            list_line.append(line)
        read.close()

        if len(list_line) == 1:
            list_split = list_line[0].split(" ")
            if list_split[0] == "paired":  # cont <- bi, cont
                if list_split[-1][:-1] == "True":
                    report.write(
                        f"Two functions show significant difference ( p < {p_interaction})\n\n"
                    )
                else:
                    report.write(
                        f"Two functions do not show significant difference ( p > {p_interaction})\n\n"
                    )

        elif len(list_line) > 1:
            for i in range(len(list_line)):
                if list_line[i].split(" ")[0] == "binary":
                    if list_line[i].split(" ")[2] == "upper:":
                        if list_line[i].split(" ")[-1][:-1] == "True":
                            report.write(
                                f"When interaction feature >= average, two functions show significant difference ( p < {p_interaction})\n\n"
                            )
                        else:
                            report.write(
                                f"When interaction feature >= average, two functions do not show significant difference ( p > {p_interaction})\n\n"
                            )

                    elif list_line[i].split(" ")[2] == "lower:":
                        if list_line[i].split(" ")[-1][:-1] == "True":
                            report.write(
                                f"When interaction feature < average, two functions show significant difference ( p < {p_interaction})\n\n"
                            )
                        else:
                            report.write(
                                f"When interaction feature < average, two functions do not show significant difference ( p > {p_interaction})\n\n"
                            )
                    else:
                        if list_line[i].split(" ")[-1][:-1] == "True":
                            report.write(
                                f'When interaction feature is {list_line[0].split(" ")[2]}, Two functions show significant difference ( p < {p_interaction})\n\n'
                            )
                        else:
                            report.write(
                                f'When interaction feature is {list_line[0].split(" ")[2]}, Two functions do not show significant difference ( p > {p_interaction})\n\n'
                            )

                elif list_line[i].split(" ")[0] == "discrete":
                    if list_line[i].split(" ")[2] == "upper:":
                        if list_line[i].split(" ")[-1][:-1] == "True":
                            report.write(
                                f"When interaction feature >= average, functions show significant difference ( p < {p_interaction})\n\n"
                            )
                        else:
                            report.write(
                                f"When interaction feature >= average, functions do not show significant difference ( p > {p_interaction})\n\n"
                            )

                    elif list_line[i].split(" ")[2] == "lower:":
                        if list_line[i].split(" ")[-1][:-1] == "True":
                            report.write(
                                f"When interaction feature < average, functions show significant difference ( p < {p_interaction})\n\n"
                            )
                        else:
                            report.write(
                                f"When interaction feature < average, functions do not show significant difference ( p > {p_interaction})\n\n"
                            )
                    else:
                        if list_line[i].split(" ")[-1][:-1] == "True":
                            report.write(
                                f'When interaction feature is {list_line[0].split(" ")[2]}, functions show significant difference ( p < {p_interaction})\n\n'
                            )
                        else:
                            report.write(
                                f'When interaction feature is {list_line[0].split(" ")[2]}, functions do not show significant difference ( p > {p_interaction})\n\n'
                            )

        if feat_type[i_t] != 1:
            if feat_type[i_i] != 1:
                pass
            else:
                pass
        else:
            if feat_type[i_i] != 1:
                pass
            else:
                pass
    report.close()

    try:
        shutil.rmtree(os.path.join(path_result, "_tmp"))
        shutil.rmtree(os.path.join(path_result, "univariate_analysis", "_tmp"))
        shutil.rmtree(os.path.join(path_result, "interactive_analysis", "_tmp"))
    except FileNotFoundError:
        pass
    print(">> Comprehension Complete")


if __name__ == "__main__":
    main()
