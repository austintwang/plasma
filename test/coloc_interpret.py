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

    data_df = pd.DataFrame.from_records(data_list)
    # print(data_df.columns.values) ####
    return data_df

def roc(ppt, ppf):
    pp = [(i, True) for i in ppt]
    pp.extend([(i, False) for i in ppf])
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
        coord = (t_num / t_tot, f_num / f_tot)
        coords.append(coord)
    # print(coords) ####

    x, y = zip(*coords)
    auroc = np.trapz(y, x=x)

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
        convert_wide=True
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
    sns.heatmap(heat_data_main, annot=True, fmt=fmt, square=True)
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
    title_base = "Mean {0}\n{1} Model, GWAS Heritability = {2:.0E}"

    sns.set(font="Roboto")

    for h in gwas_herits:
        if "full" in model_flavors:
            df_model = df.loc[
                (df["model"] == "full")
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
            model_name = "PLASMA/C-JC"
            title = title_base.format(response, model_name, h)
            result_path = os.path.join(res_dir, "full_h_{0}.svg".format(h))
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
        if "indep" in model_flavors:
            df_model = df.loc[
                (df["model"] == "indep")
                & (df["herit_gwas"] == h)
                & (df["complete"] == True)
            ]
            # print(df_model.columns.values) ####
            # print(df) ####
            # print(df.loc[
            #   (df["model"] == "indep")
            #   & (df["complete"] == True)
            # ]) ####
            # print(df_model) ####
            df_model.rename(
                columns={
                    "num_samples_gwas": var_row,
                    "num_samples_qtl": var_col,
                    "h4": response,
                }, 
                inplace=True
            )
            # print(df_model.columns.values) ####
            model_name = "PLASMA/C-J"
            title = title_base.format(response, model_name, h)
            result_path = os.path.join(res_dir, "indep_h_{0}.svg".format(h))
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
        if "ase" in model_flavors:
            df_model = df.loc[
                (df["model"] == "ase")
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
            model_name = "PLASMA/C-AS"
            title = title_base.format(response, model_name, h)
            result_path = os.path.join(res_dir, "ase_h_{0}.svg".format(h))
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
        if "ecav" in model_flavors:
            df_model = df.loc[
                (df["model"] == "ecav")
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
            model_name = "eCAVIAR"
            title = title_base.format(response, model_name, h)
            result_path = os.path.join(res_dir, "ecav_h_{0}.svg".format(h))
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
        if "eqtl" in model_flavors:
            df_model = df.loc[
                (df["model"] == "eqtl")
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
            model_name = "QTL-Only"
            title = title_base.format(response, model_name, h)
            result_path = os.path.join(res_dir, "eqtl_h_{0}.svg".format(h))
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
    title_base = "Mean {0} for Unshared Causal Markers\n{1} Model, LD Threshold = {2:.0E}"

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
            result_path = os.path.join(res_dir, "{0}_h=l_{1}.svg".format(m, l))
            make_heatmap(
                df_model, 
                var_row, 
                var_col, 
                response,
                NAMEMAP[m], 
                title, 
                result_path, 
                fmt='.2g',
                convert_wide=False
            )

if __name__ == '__main__':
    data_dir_base = "/agusevlab/awang/job_data/sim_coloc/outs/"
    res_dir_base = "/agusevlab/awang/ase_finemap_results/sim_coloc/"
    model_flavors = set(["indep", "eqtl", "ase", "ecav"])

    gwas_herits = [0.001, 0.0001]
    # interpret_shared(data_dir_base, gwas_herits, model_flavors, res_dir_base)

    ld_thresh = [0., 0.2, 0.4, 0.8, 0.95]
    interpret_corr(data_dir_base, ld_thresh, model_flavors, res_dir_base)