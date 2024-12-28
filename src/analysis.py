import os, json, argparse

from clesh import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


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
    print(">> Start Analysis")
    args = parse_args()

    with open(args.config, "r") as f:
        path_config = json.load(f)

    PATH = path_config["PATH"]

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

    avg_class = {0: "upper", 1: "lower"}
    path_load = os.path.join(PATH, "data")
    path_save = os.path.join(PATH, "clesh_results")

    data_feat = pd.read_csv(os.path.join(path_load, "features.csv"))
    data_feat = data_feat.drop(data_feat.columns[0], axis=1)
    data_shap = np.load(os.path.join(path_load, "shap.npy"))
    df_importance, rank_feat = feat_importance_ranking(data_feat, data_shap)
    print(">> Parsing Complete")

    # feature selection
    shap_ttest = feat_ttest(
        data_shap, df_importance, rank_feat, p_feature_selection, path_save=path_save
    )
    feat_tosee = feature_selection(
        shap_ttest,
        rank_feat,
        candidate_num_min,
        candidate_num_max,
        manual_num,
        path_save=path_save,
    )

    fig = shap.summary_plot(
        data_shap, data_feat, show=False, max_display=len(feat_tosee)
    )
    plt.savefig(os.path.join(path_save, "shap_summary_plot.jpg"), dpi=300)
    plt.clf()
    feat_type = feat_type_describe(data_feat=data_feat, path_save=path_save)
    np.save(os.path.join(path_save, "_tmp", "feat_type.npy"), feat_type)
    print(">> Feature Selection Complete")

    funcs_best = {}
    # univariate analysis
    for i_rank, i_t in enumerate(feat_tosee):
        name_target = data_feat.columns[i_t]
        if feat_type[i_t] != 1:
            group_nest, feat_index = discrete_target(
                data_feat=data_feat,
                data_shap=data_shap,
                idx_target=i_t,
                p_bound=p_univariate,
                bool_uni=True,
                path_save=path_save,
            )
            plt.rcParams["figure.figsize"] = (2 + len(group_nest), 5)
            fig, ax = plt.subplots()
            ax.boxplot(
                group_nest,
                flierprops={"marker": "o", "markersize": 2},
                medianprops=dict(color="black"),
            )
            ax.set_xticklabels([feat_index[idx] for idx in range(len(group_nest))])
            ax.set_xlabel(f"{name_target}")
            ax.set_ylabel("SHAP value")
            plt.tight_layout()

            plt.savefig(
                f'{make_dir(path_save+"/univariate_analysis")}/{name_target}.jpg',
                dpi=300,
            )

        elif feat_type[i_t] == 1:
            plt.rcParams["figure.figsize"] = (10, 5)
            fig, ax = plt.subplots()
            ax.set_title(data_feat.columns[i_t])
            ax.set_xlabel("feature value")
            ax.set_ylabel("SHAP value")
            ax, func_best = continuous_target(
                data_feat=data_feat,
                data_shap=data_shap,
                idx_target=i_t,
                ax=ax,
                p_bound=p_univariate,
            )
            funcs_best[i_t] = func_best
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                f'{make_dir(path_save+"/univariate_analysis")}/{name_target}.jpg',
                dpi=300,
            )
            text = open(
                f'{make_dir(path_save+"/univariate_analysis/_tmp")}/{name_target}.txt',
                "a",
            )
            text.write(f"func_best: {func_best}")
            text.close()
    print(">> Univariate Analysis Complete")

    # interaction analysis
    idx_pair = []
    for i in feat_tosee:
        idx_int = shap.utils.approximate_interactions(i, data_shap, data_feat)[0]
        idx_pair.append((i, idx_int))
    np.save(os.path.join(path_save, "_tmp", "idx_pair.npy"), idx_pair)

    for i_rank, (i_t, i_i) in enumerate(idx_pair):
        name_target = data_feat.columns[i_t]
        name_interaction = data_feat.columns[i_i]

        target_value_avg = np.average(data_feat.iloc[:, i_t])

        if feat_type[i_t] != 1:
            groups_nest = []
            feats_index = []

            if feat_type[i_i] != 1:
                for value_interaction in sorted(data_feat.iloc[:, i_i].unique()):
                    group_nest, feat_index = discrete_target(
                        data_feat=data_feat,
                        data_shap=data_shap,
                        idx_target=i_t,
                        idx_interaction=i_i,
                        value_interaction=value_interaction,
                        p_bound=p_interaction,
                        path_save=path_save,
                    )
                    plt.rcParams["figure.figsize"] = (2 + len(group_nest), 5)
                    fig, ax = plt.subplots()
                    ax.boxplot(
                        group_nest,
                        flierprops={"marker": "o", "markersize": 2},
                        medianprops=dict(color="black"),
                    )
                    ax.set_xticklabels(
                        [feat_index[idx] for idx in range(len(group_nest))]
                    )
                    ax.set_xlabel(f"{name_target}")
                    ax.set_ylabel("SHAP value")
                    plt.tight_layout()

                    plt.savefig(
                        f'{make_dir(path_save+"/interactive_analysis")}/{name_target}_{name_interaction}_{value_interaction}.jpg',
                        dpi=500,
                    )

            else:

                for key in avg_class.keys():
                    group_nest, feat_index = discrete_target(
                        data_feat=data_feat,
                        data_shap=data_shap,
                        idx_target=i_t,
                        idx_interaction=i_i,
                        class_interaction=avg_class[key],
                        p_bound=p_interaction,
                        path_save=path_save,
                    )
                    groups_nest.append(group_nest)
                    feats_index.append(feat_index)

                plt.rcParams["figure.figsize"] = (2 + len(group_nest), 5)

                feat_index_tot = []
                group_nest_tot = []
                for idx in range(len(groups_nest)):
                    feat_index_tot.extend(
                        [
                            feats_index[idx][idx2]
                            for idx2 in range(len(groups_nest[idx]))
                        ]
                    )
                    group_nest_tot.extend(groups_nest[idx])

                fig, ax = plt.subplots()
                ax.boxplot(
                    group_nest_tot,
                    flierprops={"marker": "o", "markersize": 2},
                    medianprops=dict(color="black"),
                )

                tick = np.arange(1, len(feat_index_tot) + 1, 1)
                ax.set_xticks(tick)
                ax.set_xticklabels(feat_index_tot)
                ax.set_ylabel("SHAP value")
                sec = ax.secondary_xaxis(location=0)
                ax.set_xlabel(f"{name_target}", labelpad=20)

                sec.set_xticks(
                    [tick[0], tick[int(len(tick) / 2)]], labels=["\nupper", "\nlower"]
                )
                plt.tight_layout()
                plt.savefig(
                    f'{make_dir(path_save+"/interactive_analysis")}/{name_target}_{name_interaction}.jpg',
                    bbox_inches="tight",
                    dpi=500,
                )

        else:
            func_sel = funcs_best[i_t]

            plt.rcParams["figure.figsize"] = (10, 5)
            fig, ax = plt.subplots()
            ax.set_title(
                f"target: {data_feat.columns[i_t]} (interaction: {data_feat.columns[i_i]} / curve: {func_sel})"
            )
            ax.set_xlabel("feature value")
            ax.set_ylabel("SHAP value")
            group_nest = []
            index_feat = {}
            if feat_type[i_i] != 1:
                for idx, item in enumerate(sorted(data_feat.iloc[:, i_i].unique())):
                    index_feat[item] = idx
                feat_index = {v: k for k, v in index_feat.items()}

                for idx, value_interaction in enumerate(
                    sorted(data_feat.iloc[:, i_i].unique())
                ):
                    index_feat[value_interaction] = idx
                    ax, group = continuous_target(
                        data_feat=data_feat,
                        data_shap=data_shap,
                        idx_target=i_t,
                        ax=ax,
                        func_sel=func_sel,
                        idx_interaction=i_i,
                        n_c=idx,
                        value_interaction=value_interaction,
                        p_bound=p_interaction,
                    )
                    if len(group) != 0:
                        group_nest.append(group)

            else:
                for idx, key in enumerate(avg_class.keys()):
                    index_feat[avg_class[key]] = idx
                feat_index = {v: k for k, v in index_feat.items()}

                for key in avg_class.keys():
                    ax, group = continuous_target(
                        data_feat=data_feat,
                        data_shap=data_shap,
                        idx_target=i_t,
                        ax=ax,
                        func_sel=func_sel,
                        idx_interaction=i_i,
                        n_c=key,
                        class_interaction=avg_class[key],
                        p_bound=p_interaction,
                    )
                    if len(group) != 0:
                        group_nest.append(group)

            statistic_target(
                groups=group_nest,
                feat_index=feat_index,
                p_bound=p_interaction,
                path_save=f'{make_dir(path_save+"/interactive_analysis")}',
                bool_paired=True,
                name_target=name_target,
            )
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                f'{make_dir(path_save+"/interactive_analysis")}/{name_target}_{name_interaction}.jpg',
                dpi=300,
            )
    print(">> Interaction Analysis Complete")


if __name__ == "__main__":
    main()
