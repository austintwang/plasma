import numpy as np
import os
import time
import sys
# import traceback
import pickle
# import matplotlib
# matplotlib.use('Agg')
# matplotlib.rcParams['agg.path.chunksize'] = 10000
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd


def write_loci(loci, loci_dir, out_dir, out_text_file):
    loci_info = []
    for i in loci:
        locus_path = os.path.join(loci_dir, i, "output.pickle")
        with open(locus_path, "rb") as locus_file:
            locus = pickle.load(locus_file, encoding='latin1')
        
        for j in np.nonzero(locus["causal_set_indep"])[0]:
            snp_id = locus["snp_ids"][j]
            ppa = locus["ppas_indep"][j]
            z_phi = locus["z_phi"][j]
            z_beta = locus["z_beta"][j]
            line = "{0}\t{1}\t{2}\t{3}\t{4}\n".format(i, snp_id, ppa, z_phi, z_beta)
            print(line)
            loci_info.append(line)

    with open(out_file, "w"):
        out_file.writelines(loci_info)

if __name__ == '__main__':
    loci = [
        "ENSG00000047315.10",
        "ENSG00000071082.6",
        "ENSG00000085872.10",
        "ENSG00000087470.13",
        "ENSG00000088986.6",
        "ENSG00000102900.8",
        "ENSG00000103168.12",
        "ENSG00000105401.2",
        "ENSG00000108592.12",
        "ENSG00000109519.8",
        "ENSG00000109775.6",
        "ENSG00000109920.8",
        "ENSG00000110074.6",
        "ENSG00000112576.8",
        "ENSG00000115750.12",
        "ENSG00000116016.9",
        "ENSG00000122687.13",
        "ENSG00000122965.6",
        "ENSG00000130640.9",
        "ENSG00000140854.8",
        "ENSG00000146282.13",
        "ENSG00000147601.9",
        "ENSG00000152253.4",
        "ENSG00000162851.6",
        "ENSG00000164587.7",
        "ENSG00000174780.11",
        "ENSG00000175390.8",
        "ENSG00000175582.15",
        "ENSG00000176386.4",
        "ENSG00000183207.8",
        "ENSG00000198755.6",
    ]
    loci_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"
    out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/validation"
    out_text_file = "snp_info.txt"
    write_loci(loci, loci_dir, out_dir, out_text_file)


    