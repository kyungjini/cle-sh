import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import odr, stats
from scipy.optimize import curve_fit

def rmse(pred, true):
    error = pred - true
    error = error**2
    error = np.sqrt(sum(error))
    return np.round(error, 5)


def make_dir(str_location):
    if not os.path.isdir(str_location):
        os.makedirs(str_location)

    return str_location


def feat_type_describe(data_feat, cont_bound=10, path_save=None):
    # discrete: 0, continuous: 1, binary: 2
    type_dict = {0:'discrete', 1:'continuous', 2:'binary'}
    
    n_feat = data_feat.shape[1]
    list_feat_type = np.zeros(n_feat, dtype=np.int8)
    
    text = open(make_dir(path_save+'/_tmp')+'/feat_type_describe.txt', 'w')
    text.write(f'Number of samples: {data_feat.shape[0]}\n')
    text.write(f'Number of features: {n_feat}\n\n')


    for i in range(n_feat):
        if data_feat.iloc[:, i].nunique() == 2:
            list_feat_type[i] = 2

        elif data_feat.iloc[:, i].nunique() > cont_bound:
            list_feat_type[i] = 1

    n_disc = (list_feat_type == 0).sum()
    n_cont = (list_feat_type == 1).sum()
    n_bi = (list_feat_type == 2).sum()

    text.write(f'(Binary: {n_bi}, Discrete: {n_disc}, continuous: {n_cont})\n\n')
    text.write('feature type\n')
    text.write('index: feature: type\n')
    for i in range(n_feat):
        text.write(f'{data_feat.columns[i]}: {type_dict[list_feat_type[i]]}\n')
    
    return list_feat_type


def feat_importance_ranking(data_feat, data_shap):
    # feature importance ranking
    shap_sum = np.abs(data_shap).mean(axis=0)
    df_importance = pd.DataFrame([data_feat.columns.tolist(), shap_sum.tolist()]).T
    df_importance.columns = ['column_name', 'shap_importance']
    df_importance = df_importance.sort_values('shap_importance', ascending=False)

    return df_importance, list(df_importance.index)


def feat_ttest(data_shap, df_importance, rank_feat, p_feature_selection=0.05, path_save=None):
    # SHAP t-test heatmap
    plt.tight_layout()
    
    bool_normal = True
    for i in range(len(rank_feat)):
        if stats.shapiro(data_shap[:, rank_feat[i]])[1] > p_feature_selection:
            bool_normal = False
    
    shap_ttest = np.ones((len(rank_feat), len(rank_feat)))

    if bool_normal:
        for i in range(len(rank_feat)):
            for j in range(len(rank_feat)):
                if (
                    stats.ranksums(
                        np.abs(data_shap[:, rank_feat[i]]),
                        np.abs(data_shap[:, rank_feat[j]]),
                    )[1]
                    < p_feature_selection
                ):
                    shap_ttest[i, j] = 0
    else:
        for i in range(len(rank_feat)):
            for j in range(len(rank_feat)):
                if (
                    stats.ttest_rel(
                        np.abs(data_shap[:, rank_feat[i]]),
                        np.abs(data_shap[:, rank_feat[j]]),
                    )[1]
                    < p_feature_selection
                ):
                    shap_ttest[i, j] = 0
    
    return shap_ttest


def feature_selection(shap_ttest, rank_feat, candidate_num_min=5, candidate_num_max=20, manual_num: int=0 , path_save=None):
    plt.tight_layout()
    
    text = open(make_dir(path_save+'/_tmp')+'/feature_selection.txt', 'w')
    
    feat_cnt = []
    for i in range(len(rank_feat)-1):
        if i > candidate_num_max:
            break
        if shap_ttest[i, i + 1] == 0:
            feat_cnt.append(i + 1)
    feat_cnt.append(len(rank_feat))
            
    feat_der = {}
    for i in range(len(feat_cnt) - 1):
        der = feat_cnt[i + 1] - feat_cnt[i]
        if feat_der.get(der) is None:
            feat_der[der] = [] 
        feat_der[der].append(feat_cnt[i])

    feats_rank_cut = []
    feat_der_key = sorted(feat_der, reverse=True)
    bnd = candidate_num_min
    cnt = 1
    for i in range(len(feat_der_key) - 1):
        cand = [
            item
            for item in feat_der[feat_der_key[i]]
            if (item >= bnd) & (item <= candidate_num_max)
        ]
        if len(cand) != 0:
            text.write(f'sel #{cnt}: {cand}\n')
            feats_rank_cut.extend(cand)
            bnd = feat_der[feat_der_key[i]][-1]
            cnt += 1
    if (feat_cnt[-1] >= candidate_num_min) & (feat_cnt[-1] <= candidate_num_max):
        feats_rank_cut.append(feat_cnt[-1])
        text.write(f'sel last: {feats_rank_cut[-1]}\n')
    
    plt.rcParams['figure.figsize'] = (10, 10)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(feat_cnt) + 1), feat_cnt, color='black', marker='o')
    ax.set_title('Feature Selection')
    ax.set_xticks(np.arange(1, len(feat_cnt) + 1))
    y_feat_cnt = [ i if i in feat_cnt else ' ' for i in range(1, max(feat_cnt)+1, 1)]
    ax.set_yticks(np.arange(1, feat_cnt[-1] + 1), y_feat_cnt)
    ax.grid()
    plt.tight_layout()
    plt.savefig(f'{make_dir(path_save)}/feature_selection.jpg', dpi=200)
    plt.clf()
    
    if manual_num != 0:
        text.write(f'manual rank cut: {manual_num}\n')
        text.write(f'\* manually selected')
        text.close()
        np.save(path_save+'/_tmp/rank_feat.npy', rank_feat[:manual_num])
        return rank_feat[:manual_num]
    
    elif len(feats_rank_cut) > 0:
        text.write(f'optimized rank cut: {feats_rank_cut[0]}\n')
        text.write(f'* rank cut optimized')
        text.close()
        np.save(path_save+'/_tmp/rank_feat.npy', rank_feat[:feats_rank_cut[0]])
        return rank_feat[:feats_rank_cut[0]]
    
    else:
        text.write(f'rank cut: {candidate_num_min}\n')
        text.write('* caution: rank cut optimization failed, candidate_num_min selected')
        text.close()
        np.save(path_save+'/_tmp/rank_feat.npy', rank_feat[:candidate_num_min])
        return rank_feat[:candidate_num_min]


def statistic_target(groups,
                feat_index: dict,
                p_bound: float = 0.05,
                bool_paired:bool = False,
                bool_uni:bool = False,
                path_save=None,
                name_target:str = None, 
                name_interaction:str = None, 
                value_interaction:str = None,
                class_interaction:str = None,
):
    plt.tight_layout()
    
    text = open(make_dir(f'{path_save}/_tmp')+f'/{name_target}.txt', 'a')
    
    bool_normal = True
    for idx in range(len(groups)):
        if len(groups[idx]) > 2:
            p_shapiro = stats.shapiro(groups[idx])[1] < p_bound
            if not p_shapiro:
                bool_normal = False
            
            if bool_uni == True:
                if p_shapiro:
                    if np.round(stats.ttest_1samp(groups[idx], popmean=0, alternative="less")[1], 5) < p_bound:
                        text.write(f'decrease: {feat_index[idx]}\n')
                    elif np.round(stats.ttest_1samp(groups[idx], popmean=0, alternative="greater")[1], 5) < p_bound:
                        text.write(f'increase: {feat_index[idx]}\n')
                else:
                    if np.round(stats.wilcoxon(groups[idx], alternative="less")[1], 5) < p_bound:
                        text.write(f'decrease: {feat_index[idx]}\n')
                        
                    elif np.round(stats.wilcoxon(groups[idx], alternative="less")[1], 5) < p_bound:
                        text.write(f'increase: {feat_index[idx]}\n')
        else:
                bool_normal = False
    
    if len(groups) == 2:
        if (len(groups[0]) > 2) & (len(groups[1]) > 2):
            if bool_normal:
                if not bool_paired:
                    if np.round(stats.ttest_ind(groups[0], groups[1])[1], 5) < p_bound:
                        if value_interaction is not None:
                            text.write(f'binary difference {value_interaction}: True\n')
                        elif class_interaction is not None:
                            text.write(f'binary difference {class_interaction}: True\n')
                        else:
                            text.write(f'binary difference: True\n')
                    else:
                        if value_interaction is not None:
                            text.write(f'binary difference {value_interaction}: False\n')
                        elif class_interaction is not None:
                            text.write(f'binary difference {class_interaction}: False\n')
                        else:
                            text.write(f'binary difference: False\n')

                else:
                    if np.round(stats.ttest_rel(groups[0], groups[1])[1], 5) < p_bound:
                        if value_interaction is not None:
                            text.write(f'paired binary difference {value_interaction}: True\n')
                        elif class_interaction is not None:
                            text.write(f'paired binary difference {class_interaction}: True\n')
                        else:
                            text.write(f'paired binary difference: True\n')
                    else:
                        if value_interaction is not None:
                            text.write(f'paired binary difference {value_interaction}: False\n')
                        elif class_interaction is not None:
                            text.write(f'paired binary difference {class_interaction}: False\n')
                        else:
                            text.write(f'paired binary difference: False\n')
                        
            else:
                if not bool_paired:
                    # print('< mann whitney u test >')
                    if np.round(stats.mannwhitneyu(groups[0], groups[1])[1], 5) < p_bound:
                        if value_interaction is not None:
                            text.write(f'binary difference {value_interaction}: True\n')
                        elif class_interaction is not None:
                            text.write(f'binary difference {class_interaction}: True\n')
                        else:
                            text.write(f'binary difference: True\n')
                    else:
                        if value_interaction is not None:
                            text.write(f'binary difference {value_interaction}: False\n')
                        elif class_interaction is not None:
                            text.write(f'binary difference {class_interaction}: False\n')
                        else:
                            text.write(f'binary difference: False\n')
                        
                else:
                    # print('< Wilcoxon rank sum test >')
                    if np.round(stats.ranksums(groups[0], groups[1])[1], 5) < p_bound:
                        if value_interaction is not None:
                            text.write(f'paired binary difference {value_interaction}: True\n')
                        elif class_interaction is not None:
                            text.write(f'paired binary difference {class_interaction}: True\n')
                        else:
                            text.write(f'paired binary difference: True\n')
                    else:
                        if value_interaction is not None:
                            text.write(f'paired binary difference {value_interaction}: False\n')
                        elif class_interaction is not None:
                            text.write(f'paired binary difference {class_interaction}: False\n')
                        else:
                            text.write(f'paired binary difference: False\n')
        else:
            print(f'    empty label exists!')

    elif len(groups) > 2:
        groups_new = []
        feat_index_new = {}
        for k in range(len(groups)):
            if len(groups[k]) > 1:
                groups_new.append(groups[k])
                feat_index_new[feat_index[k]] = len(groups_new)

        if len(groups_new) < 2:
            print('one or no group for the target value', end='')

        else:
            if bool_normal:
                p_disc = stats.f_oneway(*groups)[1]
                print('< ANOVA analysis >')
                print(f'    p-value: {np.round(p_disc, 5)}')
                if np.round(p_disc, 5) < p_bound:
                    if value_interaction is not None:
                        text.write(f'discrete difference {value_interaction}: True\n')
                    elif class_interaction is not None:
                        text.write(f'discrete difference {class_interaction}: True\n')
                    else:
                        text.write(f'discrete difference: True\n')
                        
                else:
                    if value_interaction is not None:
                        text.write(f'discrete difference {value_interaction}: False\n')
                    elif class_interaction is not None:
                        text.write(f'discrete difference {class_interaction}: False\n')
                    else:
                        text.write(f'discrete difference: False\n')
                
            else:
                p_disc = stats.kruskal(*groups)[1]
                print('< Kruskal-Wallis analysis >')
                print(f'    p-value: {np.round(p_disc, 5)}')
                if np.round(p_disc, 5) < p_bound:
                    if value_interaction is not None:
                        text.write(f'discrete difference {value_interaction}: True\n')
                    elif class_interaction is not None:
                        text.write(f'discrete difference {class_interaction}: True\n')
                    else:
                        text.write(f'discrete difference: True\n')
                else:
                    if value_interaction is not None:
                        text.write(f'discrete difference {value_interaction}: False\n')
                    elif class_interaction is not None:
                        text.write(f'discrete difference {class_interaction}: False\n')
                    else:
                        text.write(f'discrete difference: False\n')

            if p_disc < p_bound:
                print('< Tukey-HSD >')
                print(f'feat:index / {feat_index_new}')
                tukey = stats.tukey_hsd(*groups_new).pvalue
                for r in range(len(tukey)):
                    for c in range(r + 1):
                        tukey[r][c] = 1 if tukey[r][c] < p_bound else 0

                plt.rcParams['figure.figsize'] = (5, 5)
                sns.set_theme(style='white', font_scale=1.5)
                color_tukey = ['blue', 'red']
                cmap = LinearSegmentedColormap.from_list('Custom', colors=color_tukey, N=2)
                
                mask = np.triu(np.ones_like(tukey))
                ax = sns.heatmap(
                            tukey,
                            lw=1,
                            linecolor='white',
                            cmap=cmap,
                            mask=mask, 
                            xticklabels=list(feat_index_new.keys()),
                            yticklabels=list(feat_index_new.keys())
                        )
                colorbar = ax.collections[0].colorbar
                colorbar.set_ticks([0, 1])
                colorbar.set_ticklabels([f'p > {p_bound}', f'p < {p_bound}'])
                ax.set_title('Tukey-HSD')
                ax.set_xlabel('feature value')
                ax.set_ylabel('feature value')
                _, labels = plt.yticks()
                plt.setp(labels, rotation=0)
                plt.tight_layout()

                if value_interaction is not None:
                    plt.savefig(f'{path_save}/tukey_{name_target}_{name_interaction}_{value_interaction}.jpg', dpi=50)
                elif class_interaction is not None:
                    plt.savefig(f'{path_save}/tukey_{name_target}_{name_interaction}_{class_interaction}.jpg', dpi=50)
                else:
                    plt.savefig(f'{path_save}/tukey_{name_target}.jpg', dpi=50)
                    
                plt.clf()

            print('\n')

    else:
        pass
    text.close()


def discrete_target(
    data_feat,
    data_shap,
    idx_target: int,
    idx_interaction: int = None,
    value_interaction: float = None,
    class_interaction: str = None,
    p_bound: float = 0.05,
    bool_uni: bool = False,
    path_save: str = None
):    
    name_target = data_feat.columns[idx_target]
    if idx_interaction is not None:
        name_interaction = data_feat.columns[idx_interaction]
        
        if class_interaction is not None:
            value_interaction_avg = np.average(data_feat.iloc[:, idx_interaction])

            ser_bool = data_feat[name_interaction] > value_interaction_avg
            ser_bool = ser_bool if class_interaction == 'upper' else ~ser_bool
            feat_target = data_feat[ser_bool].iloc[:, idx_target]
            idxs = data_feat[ser_bool].index
        else:
            feat_target = data_feat[data_feat[name_interaction] == value_interaction].iloc[:, idx_target]
            idxs = data_feat[data_feat[name_interaction] == value_interaction].index

    else:
        feat_target = data_feat.iloc[:, idx_target]
        idxs = np.arange(0, len(feat_target), 1)

    group_nest = [[] for _ in range(feat_target.nunique())]
    index_feat = {}

    for idx, item in enumerate(sorted(feat_target.unique())):
        index_feat[item] = idx

    feat_index = {v: k for k, v in index_feat.items()}


    for idx in idxs:
        group_nest[index_feat[feat_target[idx]]].append(data_shap[idx][idx_target])
        
    if idx_interaction is not None:
        if class_interaction is not None:
            statistic_target(groups=group_nest, feat_index=feat_index, p_bound=p_bound, path_save=f'{make_dir(path_save+"/interactive_analysis")}', name_target=name_target, name_interaction=name_interaction, class_interaction=class_interaction)
        else:
            statistic_target(groups=group_nest, feat_index=feat_index, p_bound=p_bound, path_save=f'{make_dir(path_save+"/interactive_analysis")}', name_target=name_target, name_interaction=name_interaction, value_interaction=value_interaction)
    else:
        statistic_target(groups=group_nest, feat_index=feat_index, p_bound=p_bound, bool_uni=bool_uni, path_save=f'{make_dir(path_save+"/univariate_analysis")}', name_target=name_target)
    
    return group_nest, feat_index


def continuous_target(
    data_feat,
    data_shap,
    idx_target: int,
    ax,
    func_sel: list = None,
    idx_interaction: int = None,
    n_c: int = None,
    value_interaction: float = None,
    class_interaction: str = None,
    p_bound: float = 0.05,
):
    colorset = [
        'red',
        'blue',
        'green',
        'dodgerblue',
        'olive',
        'darkviolet',
        'orangered',
        'green',
        'deeppink',
    ]
    def linear(x, b0, b1):
        return b0 + (b1 * x)

    def linear_wrapper(params, x):
        return linear(x, *params)

    def quadratic(x, b0, b1, b2):
        return b0 + (b1 * x) + b2 * (x**2)

    def quadratic_wrapper(params, x):
        return quadratic(x, *params)

    def sigmoid(x, L, x0, b, k):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return y

    def sigmoid_wrapper(params, x):
        return sigmoid(x, *params)
    
    plt.tight_layout()
    if idx_interaction is not None:
        name_interaction = data_feat.columns[idx_interaction]
        if class_interaction is not None:
            value_interaction_avg = np.average(data_feat.iloc[:, idx_interaction])

            ser_bool = data_feat[name_interaction] > value_interaction_avg
            ser_bool = ser_bool if class_interaction == 'upper' else ~ser_bool
            idxs = data_feat[ser_bool].index
        else:
            idxs = data_feat[data_feat[name_interaction] == value_interaction].index

        x = np.array(data_feat.iloc[:, idx_target]).take(idxs)
        y = data_shap[:, idx_target].take(idxs)
        funcs = [func_sel]
    else:
        x = np.array(data_feat.iloc[:, idx_target])
        y = data_shap[:, idx_target]
        funcs = ['linear', 'quadratic', 'sigmoid']

    p0 = [max(y), np.median(x), 1, min(y)]

    xq3, xq1 = np.quantile(x, [0.75, 0.25])
    xiqr = xq3 - xq1

    idx_in = [(itemx <= xq3 + 1.5 * xiqr) & (itemx >= xq1 - 1.5 * xiqr) for itemx in x]
    x_in = x[idx_in]
    y_in = y[idx_in]

    max_x, min_x = np.max(x_in), np.min(x_in)
    grid_x = np.linspace(min_x, max_x, 100)

    if idx_interaction is not None:
        ax.scatter(x_in, y_in, alpha=0.5, s=5, facecolors='none', edgecolors=colorset[n_c])
    else:
        ax.scatter(x_in, y_in, alpha=0.5, s=5, facecolors='none', edgecolors='black')
    
    func_best = 'None'
    y_pred_grid_best = None
    group = None
    error_bound = 10000
    for n, func in enumerate(funcs):
        try:
            popt_cand = []
            error_func_cand = []
            if func == 'sigmoid':
                for method in ['trf', 'dogbox', 'lm']:
                    try:
                        popt, _ = curve_fit(locals()[func], x, y, p0, method=method, maxfev=2000)
                        popt_cand.append(popt)
                        y_pred_real = locals()[func](x, *popt)
                        error_func_cand.append(rmse(y, y_pred_real))
                    except:
                        pass
            else:
                for method in ['trf', 'dogbox', 'lm']:
                    try:
                        popt, _ = curve_fit(locals()[func], x, y, method=method, maxfev=2000)
                        popt_cand.append(popt)
                        y_pred_real = locals()[func](x, *popt)
                        error_func_cand.append(rmse(y, y_pred_real))
                    except:
                        pass

            error_func = min(error_func_cand)
            popt = popt_cand[error_func_cand.index(error_func)]
            
            model = odr.Model(locals()[func + '_wrapper'])
            data = odr.Data(x, y)
            myodr = odr.ODR(data, model, beta0=popt, maxit=0)

            myodr.set_job(fit_type=2)

            param_stats = myodr.run()
            df_e = len(x) - len(popt)
            tstat_beta = popt / param_stats.sd_beta
            pstat_beta = (1.0 - stats.t.cdf(np.abs(tstat_beta), df_e)) * 2.0

            pvalue_func = pstat_beta[-1]
            y_pred_grid = locals()[func](grid_x, *popt)
            
            if idx_interaction is not None:
                if class_interaction is not None:
                    label = class_interaction
                else:
                    label = value_interaction
                if pvalue_func < p_bound:
                    ax.plot(
                        grid_x,
                        y_pred_grid,
                        alpha=1,
                        c=colorset[n_c],
                        label=label,
                    )
                    every_x = np.array(data_feat.iloc[:, idx_target])
                    group = locals()[func](every_x, *popt)

            else:
                label = func
                # ax.plot(grid_x, y_pred_grid, alpha=1, c=colorset[n], label=func)
                if (error_func < error_bound) & (pvalue_func < p_bound):
                    error_bound = error_func
                    func_best = func
                    y_pred_grid_best = y_pred_grid
        except:
            pass
        
    if func_best != 'None':
        ax.plot(grid_x, y_pred_grid_best, alpha=1, c='red', label=func_best)

    if idx_interaction is not None:
        if group is None:
            group = []
        return ax, group
    else:
        return ax, func_best
    
    
