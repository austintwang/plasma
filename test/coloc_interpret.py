import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import pickle as pickle
except ImportError:
    import pickle

# COLORMAP = {
#     "full": pal[6],
#     "indep": pal[0],
#     "ase": pal[4],
#     "acav": pal[2],
#     "eqtl": pal[7],
#     "cav": pal[5],
#     "rasq": pal[1],
#     "fmb": pal[3],
# }

NAMEMAP = {
    "full": "PLASMA/C-JC",
    "indep": "PLASMA/C-J",
    "ase": "PLASMA/C-AS",
    "eqtl": "QTL-Only",
    "ecav": "eCAVIAR",
}

def load_data(data_dir):
    # print(os.listdir(data_dir)) ####
    filenames = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if i.endswith(".pickle")]
    data_list = []
    for i in filenames:
        # print(i) ####
        with open(i, "rb") as data_file:
            data = pickle.load(data_file)
        data_list.extend(data)

    print(data_list[0]) ####
    data_df = pd.DataFrame.from_records(data_list)
    print(data_df.columns.values) ####
    return data_df

def roc(ppt, ppf):
    pp = [(i, True) for i in np.nan_to_num(ppt)]
    pp.extend([(i, False) for i in np.nan_to_num(ppf)])
    pp.sort(reverse=True)
    # print(pp) ####

    t_tot = np.size(ppt)
    f_tot = np.size(ppf)
    
    coords = [(0.,0.),]
    t_num = 0
    f_num = 0
    for i in pp:
        if i[1]:
            t_num += 1
        else:
            f_num += 1
        coord = (f_num / f_tot, t_num / t_tot)
        coords.append(coord)
    # print(coords) ####

    x, y = zip(*coords)
    auroc = np.trapz(y, x=x)

    # if auroc == 1:
    #     print(pp) ####

    return auroc

def make_heatmap(
        df_main,
        # df_rmargin,
        # df_cmargin, 
        # df_tmargin,
        var_row, 
        var_col, 
        response, 
        model_name, 
        title, 
        result_path, 
        aggfunc="mean",
        fmt='.2g',
        convert_wide=True,
        heatmap_kwargs={}
    ):
    if convert_wide:
        heat_data_main = pd.pivot_table(
            df_main, 
            values=response, 
            index=var_row, 
            columns=var_col, 
            aggfunc=aggfunc
        )
    else:
        heat_data_main = df_main
    # heat_data_rmargin = pd.pivot_table(
    #   df_rmargin, 
    #   values=response, 
    #   index=var_row, 
    #   columns=var_col, 
    #   aggfunc=aggfunc
    # )
    # heat_data_cmargin = pd.pivot_table(
    #   df_cmargin, 
    #   values=response, 
    #   index=var_row, 
    #   columns=var_col, 
    #   aggfunc=aggfunc
    # )
    # heat_data_tmargin = pd.pivot_table(
    #   df_tmargin, 
    #   values=response, 
    #   index=var_row, 
    #   columns=var_col, 
    #   aggfunc=aggfunc
    # )
    # print(df) ####
    # print(heat_data) ####
    plt.subplots(figsize=(5.1,4.3))
    ax = sns.heatmap(heat_data_main, annot=True, fmt=fmt, square=True, cbar=False, vmin=0., vmax=1., **heatmap_kwargs)
    ax.set_xticklabels([int(float(i.get_text())) for i in ax.get_xticklabels()])
    ax.set_yticklabels([int(float(i.get_text())) for i in ax.get_yticklabels()])
    plt.yticks(rotation=0) 
    plt.title(title)
    plt.savefig(result_path)
    plt.clf()

def interpret_shared(
        data_dir_base, 
        gwas_herits, 
        model_flavors,
        res_dir_base
    ):
    data_dir = os.path.join(data_dir_base, "shared")
    df = load_data(data_dir)

    res_dir = os.path.join(res_dir_base, "shared")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    var_row = "GWAS Sample Size"
    var_col = "QTL Sample Size"
    response = "Colocalization Score (PP4)"
    title_base = "{1}, GWAS Heritability = {2:.1E}"

    sns.set(font="Roboto")

    for h in gwas_herits:
        for m in model_flavors:
            df_model = df.loc[
                (df["model"] == m)
                & (df["herit_gwas"] == h)
                & (df["complete"] == True)
            ]
            df_model.rename(
                columns={
                    "num_samples_gwas": var_row,
                    "num_samples_qtl": var_col,
                    "h4": response,
                }, 
                inplace=True
            )
            model_name = NAMEMAP[m]
            title = title_base.format(response, model_name, h)
            result_path = os.path.join(res_dir, "{0}_h_{1}.svg".format(m, h))
            make_heatmap(
                df_model, 
                var_row, 
                var_col, 
                response, 
                model_name, 
                title, 
                result_path, 
                aggfunc="mean",
                fmt='.2g'
            )


def interpret_shared_xpop(
        data_dir_base, 
        populations, 
        model_flavors,
        res_dir_base
    ):
    data_dir = os.path.join(data_dir_base, "shared_xpop")
    df = load_data(data_dir)

    res_dir = os.path.join(res_dir_base, "shared_xpop")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    var_row = "GWAS Sample Size"
    var_col = "QTL Sample Size"
    response = "Colocalization Score (PP4)"
    title_base = "{1}, {2} QTL, {3} GWAS"

    sns.set(font="Roboto")

    for i, p in enumerate(populations):
        for m in model_flavors:
            df_model = df.loc[
                (df["model"] == m)
                & (df["res_set"] == i)
                & (df["complete"] == True)
                & (df["h4"] != np.nan)
            ]
            df_model.rename(
                columns={
                    "num_samples_gwas": var_row,
                    "num_samples_qtl": var_col,
                    "h4": response,
                }, 
                inplace=True
            )
            model_name = NAMEMAP[m]
            title = title_base.format(response, model_name, p[0], p[1])
            result_path = os.path.join(res_dir, "{0}_q_{1}_g_{2}.svg".format(m, p[0], p[1]))
            make_heatmap(
                df_model, 
                var_row, 
                var_col, 
                response, 
                model_name, 
                title, 
                result_path, 
                aggfunc="mean",
                fmt='.2g'
            )

def interpret_shared_meta(
        data_dir_base, 
        comparisons, 
        model_flavors,
        res_dir_base
    ):
    data_dir = os.path.join(data_dir_base, "shared_meta")
    df = load_data(data_dir)

    res_dir = os.path.join(res_dir_base, "shared_meta")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    var_row = "GWAS Sample Size"
    var_col = "QTL Sample Size"
    response = "Meta-Colocalization Score"
    title_base = "{1}, {2} and {3} Meta-Analysis"

    sns.set(font="Roboto")

    for i, p in comparisons:
        print(i, p) ####
        for m in model_flavors:
            df_1 = df.loc[
                (df["model"] == m)
                & (df["res_set"] == i[0])
                & (df["complete"] == True)
            ]
            df_1.rename(
                columns={
                    "num_samples_gwas": var_row,
                    "num_samples_qtl": var_col,
                }, 
                inplace=True
            )
            df_2 = df.loc[
                (df["model"] == m)
                & (df["res_set"] == i[1])
                & (df["complete"] == True)
            ]
            df_2.rename(
                columns={
                    "num_samples_gwas": var_row,
                    "num_samples_qtl": var_col,
                }, 
                inplace=True
            )
            df_model = calc_meta(df_1, df_2, var_row, var_col)
            title = title_base.format(response, model_name, p[0], p[1])
            result_path = os.path.join(res_dir, "{0}_m_{1}_{2}.svg".format(m, p[0], p[1]))
            make_heatmap(
                df_model, 
                var_row, 
                var_col, 
                response,
                NAMEMAP[m], 
                title, 
                result_path, 
                fmt='.2g',
                convert_wide=False,
                heatmap_kwargs={"center": 0.5}
            )

def calc_meta(df_1, df_2, var_row, var_col):
    struct = pd.pivot_table(
        df_1, 
        values="h4", 
        index=var_row, 
        columns=var_col, 
    )

    cols = struct.columns.values
    rows = struct.index.values

    for r in rows:
        for c in cols:
            clpp_1 = df_1.loc[
                (df_1[var_row] == r) & (df_1[var_col] == c),
                ["clpps", "locus_pos"]
            ].sort_values("locus_pos").loc[:, "clpps"].values().flatten()
            clpp_2 = df_2.loc[
                (df_2[var_row] == r) & (df_2[var_col] == c),
                ["clpps", "locus_pos"]
            ].sort_values("locus_pos").loc[:, "clpps"].values().flatten()

            clpp_meta = np.mean([i.dot(j) for i, j in zip(clpp_1, clpp_2)])
            struct.loc[r, c] = clpp_meta
            print(clpp_meta) ####

    return struct


def calc_rocs(df_neg, df_pos, var_row, var_col, response):
    struct = pd.pivot_table(
        df_neg, 
        values=response, 
        index=var_row, 
        columns=var_col, 
    )

    cols = struct.columns.values
    rows = struct.index.values

    for r in rows:
        for c in cols:
            ppt = df_pos.loc[
                (df_pos[var_row] == r) & (df_pos[var_col] == c),
                [response]
            ].values.flatten()
            ppf = df_neg.loc[
                (df_neg[var_row] == r) & (df_neg[var_col] == c),
                [response]
            ].values.flatten()
            # print(ppt) ####
            # print(ppf) ####

            auroc = roc(ppt, ppf)
            struct.loc[r, c] = auroc
            print(auroc) ####

    return struct

def interpret_corr(
        data_dir_base, 
        ld_thresh, 
        model_flavors,
        res_dir_base
    ):
    data_dir = os.path.join(data_dir_base, "corr")
    df = load_data(data_dir)

    data_dir_ctrl = os.path.join(data_dir_base, "shared")
    df_ctrl = load_data(data_dir_ctrl)

    res_dir = os.path.join(res_dir_base, "corr")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    var_row = "GWAS Sample Size"
    var_col = "QTL Sample Size"
    response = "Area Under ROC"
    title_base = "{1}, LD Threshold = {2}"

    sns.set(font="Roboto")

    for m in model_flavors:
        df_pos = df_ctrl.loc[
            (df["model"] == m)
            & (df["herit_gwas"] == 0.001)
            & (df["complete"] == True)
        ]
        df_pos.rename(
            columns={
                "num_samples_gwas": var_row,
                "num_samples_qtl": var_col,
            }, 
            inplace=True
        )
        for l in ld_thresh:
            df_neg = df.loc[
                (df["model"] == m)
                & (df["corr_thresh"] == l)
                & (df["complete"] == True)
            ]
            df_neg.rename(
                columns={
                    "num_samples_gwas": var_row,
                    "num_samples_qtl": var_col,
                }, 
                inplace=True
            )
            df_model = calc_rocs(df_neg, df_pos, var_row, var_col, "h4")
            title = title_base.format(response, NAMEMAP[m], l)
            result_path = os.path.join(res_dir, "{0}_l_{1}.svg".format(m, l))
            make_heatmap(
                df_model, 
                var_row, 
                var_col, 
                response,
                NAMEMAP[m], 
                title, 
                result_path, 
                fmt='.2g',
                convert_wide=False,
                heatmap_kwargs={"center": 0.5}
            )

if __name__ == '__main__':
    data_dir_base = "/agusevlab/awang/job_data/sim_coloc/outs/"
    res_dir_base = "/agusevlab/awang/ase_finemap_results/sim_coloc/"
    model_flavors = set(["indep", "eqtl", "ase", "ecav"])

    # gwas_herits = [0.001, 0.0001]
    # interpret_shared(data_dir_base, gwas_herits, model_flavors, res_dir_base)

    # ld_thresh = [0., 0.2, 0.4, 0.8, 0.95]
    # interpret_corr(data_dir_base, ld_thresh, model_flavors, res_dir_base)

    # populations = [("EUR", "AFR"), ("AFR", "EUR"), ("EUR", "EUR"), ("AFR", "AFR")]
    # interpret_shared_xpop(data_dir_base, populations, model_flavors, res_dir_base)

    comparisons = [((0, 1), ("AFR", "EUR")), ((0, 2), ("EUR", "EUR")), ((1, 3), ("AFR", "AFR"))]
    interpret_shared_meta(data_dir_base, comparisons, model_flavors, res_dir_base)





