import os
import pickle
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker 
import pandas as pd 
import scipy.stats

import pybedtools

class SnpMismatchError(Exception):
    pass

def region_plotter(regions, bounds):
    def region_plot(*args, **kwargs):
        for p, q in regions:
            if p < bounds[0]:
                start = bounds[0]
            else:
                start = p
            if q > bounds[1]:
                end = bounds[1]
            else:
                end = q
            plt.axvspan(start, end, facecolor='k', linewidth=0, alpha=0.1)

    return region_plot

def plot_manhattan(pp_df, gene_name, out_dir, regions, bounds):
    sns.set(style="ticks", font="Roboto")

    pal = sns.xkcd_palette(["silver", "slate", "blood red"])

    g = sns.FacetGrid(
        pp_df, 
        row="Model", 
        hue="Causal",
        hue_kws={"marker":["o", "o", "D"]},
        palette=pal,
        margin_titles=True, 
        height=1.7, 
        aspect=3
    )

    g.map(region_plotter(regions, bounds))

    g.map(
        sns.scatterplot, 
        "Position", 
        "-log10 p-Value",
        # size="Causal", 
        legend=False,
        # color=".3", 
        linewidth=0,
        hue_order=[2, 1, 0],
        # sizes={0:9, 1:12},
        s=9
    )

    x_formatter = matplotlib.ticker.ScalarFormatter()
    for i, ax in enumerate(g.fig.axes): 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.xaxis.set_major_formatter(x_formatter)
    
    plt.subplots_adjust(top=0.9, bottom = 0.13, right = 0.96)
    g.fig.suptitle("Association Statistics for {0}".format(gene_name))
    plt.savefig(os.path.join(out_dir, "manhattan_{0}.svg".format(gene_name)))
    plt.clf()

def read_genes(list_path):
    gene_list = []
    with open(list_path) as list_file:
        for l in list_file:
            entries = l.strip().split()
            gene_list.append([entries[0], entries[2]])
    return gene_list

def analyze_locus(res_path, gene_name, annot_path, snp_filter, out_dir):
    with open(os.path.join(res_path, "output.pickle"), "rb") as res_file:
        result = pickle.load(res_file, encoding='latin1')
    with open(os.path.join(res_path, "in_data.pickle"), "rb") as inp_file:
        inputs = pickle.load(inp_file, encoding='latin1')

    pp_lst = []

    snps_in_filter = [ind for ind, val in enumerate(inputs["snp_ids"]) if val in snp_filter]
    snp_ids = inputs["snp_ids"][snps_in_filter]
    snp_pos = inputs["snp_pos"][snps_in_filter]

    llim = snp_pos[0]
    ulim = snp_pos[-1]

    cset_plasma = result["causal_set_indep"]
    cset_finemap = result["causal_set_fmb"]

    ppas_plasma = result["ppas_indep"]
    ppas_finemap = result["ppas_fmb"]

    # print(len(cset_plasma)) ####
    # print(len(ppas_plasma)) ####
    # print(len(result["informative_snps"])) ####
    # print(len(inputs["snp_ids"])) ####

    informative_snps = result["informative_snps"]

    print(len(informative_snps))
    print(len(snp_ids))

    if len(informative_snps) != len(snp_ids):
        raise SnpMismatchError

    z_phi = np.full(np.shape(informative_snps), 0.)
    np.put(z_phi, informative_snps, result["z_phi"])
    for i, z in enumerate(z_phi):
        l = -np.log10(scipy.stats.norm.sf(abs(z))*2)
        if all([cset_plasma[i] == 1, ppas_plasma[i] != np.nan, z > 0.]):
            causal = 1
        else:
            causal = 0
        if llim <= snp_pos[i] <= ulim:
            info = [snp_pos[i], l, "PLASMA", causal]
            pp_lst.append(info)

    z_beta = np.full(np.shape(informative_snps), 0.)
    np.put(z_beta, informative_snps, result["z_beta"])
    for i, z in enumerate(z_beta):
        l = -np.log10(scipy.stats.norm.sf(abs(z))*2)
        if all([cset_finemap[i] == 1, ppas_finemap[i] != np.nan, z > 0.]):
            causal = 1
        else:
            causal = 0
        if llim <= snp_pos[i] <= ulim:
            info = [snp_pos[i], l, "FINEMAP", causal]
            pp_lst.append(info)

    region_start = snp_pos[0]
    region_end = snp_pos[-1] + 1
    chromosome = "chr{0}".format(inputs["chr"])

    pp_cols = [
        "Position", 
        "-log10 p-Value", 
        "Model", 
        "Causal"
    ]

    pp_df = pd.DataFrame(pp_lst, columns=pp_cols)

    bounds = (llim, ulim)

    reg = "{0}\t{1}\t{2}".format(chromosome, llim, ulim)
    reg = pybedtools.BedTool(reg, from_string=True)
    ann = pybedtools.BedTool(annot_path)
    features = ann.intersect(reg)

    regions = []
    for f in features:
        regions.append((f.start, f.stop,))

    plot_manhattan(pp_df, gene_name, out_dir, regions, bounds)

    markers = []
    for ind, val in enumerate(cset_plasma):
        if val == 1 and any([(f.start <= snp_pos[ind] <= f.end) for f in features]):
            marker_data = [
                gene_name,
                chromosome,
                snp_ids[ind],
                ppas_plasma[ind],
                z_phi[ind],
                ppas_finemap[ind],
                z_beta[ind]
            ]
            markers.append(marker_data)

    return markers

def analyze_list(res_path_base, list_path, annot_path, filter_path, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(filter_path, "rb") as filter_file:
        snp_filter = pickle.load(filter_file)

    markers_list = []
    err_list = []

    gene_list = read_genes(list_path)
    for gene_name, gene_id in gene_list:
        res_path_wildcard = os.path.join(res_path_base, gene_id + ".*")
        res_path_matches = glob.glob(res_path_wildcard)
        if len(res_path_matches) != 1:
            print(gene_name, gene_id)
            print(res_path_matches)
            err_list.append("{0}\t{1}\t{2}\n".format(gene_name, gene_id, len(res_path_matches)))
            continue
        res_path = res_path_matches[0]
        try:
            locus_data = analyze_locus(res_path, gene_name, annot_path, snp_filter, out_dir)
        except SnpMismatchError:
            err_list.append("{0}\t{1}\t{2}\n".format(gene_name, gene_id, "data_error"))
            continue
        markers_list.extend(locus_data)

    markers_cols = [
        "Gene_Name",
        "Chromosome",
        "RSID",
        "PLASMA_PIP",
        "AS_Z",
        "FINEMAP_PIP",
        "QTL_Z"
    ]
    markers_df = pd.DataFrame(markers_list, columns=markers_cols)
    out_path = os.path.join(out_dir, "markers.txt")
    markers_df.to_csv(path_or_buf=out_path, index=False, sep="\t")

    err_path = os.path.join(out_dir, "not_found.txt")
    with open(err_path, "w") as err_file:
        err_file.writelines(err_list)


if __name__ == '__main__':
    res_path_base = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"
    val_path = "/agusevlab/awang/job_data/validation"
    annot_path = os.path.join(val_path, "786O_H3k27ac_merged_hg19.bed")
    list_path = os.path.join(val_path, "RCC.dep1.genes")
    out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/validation"
    filter_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/snp_filters/1KG_SNPs.pickle"

    analyze_list(res_path_base, list_path, annot_path, filter_path, out_dir)



