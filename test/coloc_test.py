#!/usr/bin/env python3

import os
import random
import traceback
import pickle
import signal
from contextlib import contextmanager
# import gc

import numpy as np
import pandas as pd

if __name__ == '__main__' and __package__ is None:
    __package__ = 'test'
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.insert(0, "/agusevlab/awang/plasma")

from . import Finemap, LocusSimulator, EvalECaviar

NAMEMAP = {
    "full": "PLASMA/C-JC",
    "indep": "PLASMA/C-J",
    "ase": "PLASMA/C-AS",
    "eqtl": "QTL-Only",
    "ecav": "eCAVIAR",
}

CHROM_LENS = [
    248956422,
    242193529,
    198295559,
    190214555,  
    181538259,
    170805979,
    159345973,
    145138636,
    138394717,
    133797422,
    135086622,
    133275309,
    114364328,
    107043718,
    101991189,          
    90338345,
    83257441,
    80373285,
    58617616,
    64444167,
    46709983,
    50818468,
]

class TimeoutException(Exception): 
    pass

class DummyFinemap(Finemap):
    def __init__(self, ppas, causal_set):
        self.ppas = ppas
        self.causal_set = causal_set

    def get_ppas(self):
        return self.ppas

    def get_causal_set(self, confidence):
        return self.causal_set

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException()
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def draw_region(vcf_dir, vcf_name_template):
    regions = [i - 1000000 for i in CHROM_LENS]
    weights = np.array(regions) / np.sum(regions)
    chrom_num = np.random.choice(list(range(1, 23)), p=weights)
    chrom = "chr{0}".format(chrom_num)
    vcf_path = os.path.join(vcf_dir, vcf_name_template.format(chrom))
    start = random.randrange(0, regions[chrom_num-1])

    return chrom, chrom_num, start, vcf_path

def sim_shared_causal(vcf_dir, vcf_name_template, sample_filters, snp_filter, params):
    while True:
        try:
            # if params.get("specify_region", False):
            #   chrom = params["chrom"]
            #   chrom_num = params["chrom_num"]
            #   start = params["start"]
            #   vcf_path = params["vcf_path"]
            # else:
            chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)
            # print("a") ####
            # gc.collect()
            # print("b") ####
            with time_limit(100):
                locus = LocusSimulator(
                    vcf_path, 
                    chrom_num, 
                    start, 
                    params["num_causal"],
                    region_size=params["region_size"],
                    max_snps=params["max_snps"],
                    sample_filter=sample_filters[0],
                    snp_filter=snp_filter,
                    maf_thresh=params["maf_thresh"]
                )
        except (ValueError, TimeoutException):
            continue
        if locus.snp_count >= 10:
            break

    causal_config_qtl = locus.causal_config
    causal_config_gwas = locus.causal_config

    qtl_data = locus.sim_asqtl(
        params["num_samples_qtl"],
        params["coverage"],
        params["std_al_dev"],
        params["herit_qtl"],
        params["herit_as"],
        params["overdispersion"],
    )

    gwas_data = locus.sim_gwas(
        params["num_samples_gwas"],
        params["herit_gwas"],
    )

    return locus, qtl_data, gwas_data, causal_config_qtl, causal_config_gwas

def sim_shared_causal_xpop(vcf_dir, vcf_name_template, sample_filters, snp_filter, params):
    while True:
        try:
            # if params.get("specify_region", False):
            #   chrom = params["chrom"]
            #   chrom_num = params["chrom_num"]
            #   start = params["start"]
            #   vcf_path = params["vcf_path"]
            # else:
            chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)
            # print("a") ####
            # gc.collect()
            # print("b") ####
            with time_limit(100):
                locus_1 = LocusSimulator(
                    vcf_path, 
                    chrom_num, 
                    start, 
                    params["num_causal"],
                    region_size=params["region_size"],
                    max_snps=params["max_snps"],
                    sample_filter=sample_filters[0],
                    snp_filter=snp_filter,
                    maf_thresh=params["maf_thresh"]
                )
                locus_2 = LocusSimulator(
                    vcf_path, 
                    chrom_num, 
                    start, 
                    params["num_causal"],
                    region_size=params["region_size"],
                    max_snps=params["max_snps"],
                    sample_filter=sample_filters[1],
                    snp_filter=snp_filter,
                    maf_thresh=params["maf_thresh"]
                )
        except (ValueError, TimeoutException):
            continue
        if locus_1.snp_count >= 10 and locus_1.snp_count == locus_2.snp_count:
            break

    causal_inds = np.random.choice(locus_1.snp_count, params["num_causal"], replace=False)
    causal_config = np.zeros(locus_1.snp_count)
    np.put(causal_config, causal_inds, 1)

    qtl_data_1 = locus_1.sim_asqtl(
        params["num_samples_qtl"],
        params["coverage"],
        params["std_al_dev"],
        params["herit_qtl"],
        params["herit_as"],
        params["overdispersion"],
        causal_override=causal_config
    )

    gwas_data_1 = locus_1.sim_gwas(
        params["num_samples_gwas"],
        params["herit_gwas"],
        causal_override=causal_config
    )

    qtl_data_2 = locus_2.sim_asqtl(
        params["num_samples_qtl"],
        params["coverage"],
        params["std_al_dev"],
        params["herit_qtl"],
        params["herit_as"],
        params["overdispersion"],
        causal_override=causal_config
    )

    gwas_data_2 = locus_2.sim_gwas(
        params["num_samples_gwas"],
        params["herit_gwas"],
        causal_override=causal_config
    )

    return_vals = []
    return_vals.append((locus_1, qtl_data_1, gwas_data_2, causal_config, causal_config,),)
    return_vals.append((locus_2, qtl_data_2, gwas_data_1, causal_config, causal_config,),)
    return_vals.append((locus_1, qtl_data_1, gwas_data_1, causal_config, causal_config,),)
    return_vals.append((locus_2, qtl_data_2, gwas_data_2, causal_config, causal_config,),)

    return return_vals

def sim_shared_causal_meta(vcf_dir, vcf_name_template, sample_filters, snp_filter, params):
    while True:
        try:
            # if params.get("specify_region", False):
            #   chrom = params["chrom"]
            #   chrom_num = params["chrom_num"]
            #   start = params["start"]
            #   vcf_path = params["vcf_path"]
            # else:
            chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)
            # print("a") ####
            # gc.collect()
            # print("b") ####
            with time_limit(100):
                loci_p1 = [None, None]
                for i in range(2):
                    loci_p1[i] = LocusSimulator(
                        vcf_path, 
                        chrom_num, 
                        start, 
                        params["num_causal"],
                        region_size=params["region_size"],
                        max_snps=params["max_snps"],
                        sample_filter=sample_filters[0],
                        snp_filter=snp_filter,
                        maf_thresh=params["maf_thresh"]
                    )
                loci_p2 = [None, None]
                for i in range(2):
                    loci_p2[i] = LocusSimulator(
                        vcf_path, 
                        chrom_num, 
                        start, 
                        params["num_causal"],
                        region_size=params["region_size"],
                        max_snps=params["max_snps"],
                        sample_filter=sample_filters[1],
                        snp_filter=snp_filter,
                        maf_thresh=params["maf_thresh"]
                    )

        snp_counts = [loci_p1[0].snp_count, loci_p1[1].snp_count, loci_p2[0].snp_count, loci_p2[1].snp_count]
        except (ValueError, TimeoutException):
            continue
        if snp_counts[0] >= 10 and snp_counts.count(snp_counts[0]) == len(snp_counts):
            break

    causal_inds = np.random.choice(locus_1.snp_count, params["num_causal"], replace=False)
    causal_config = np.zeros(locus_1.snp_count)
    np.put(causal_config, causal_inds, 1)

    qtl_data_p1 = [None, None]
    gwas_data_p1 = [None, None]
    for i in range(2):
        qtl_data_p1[i] = loci_p1[i].sim_asqtl(
            params["num_samples_qtl"],
            params["coverage"],
            params["std_al_dev"],
            params["herit_qtl"],
            params["herit_as"],
            params["overdispersion"],
            causal_override=causal_config
        )
        gwas_data_p1[i] = loci_p1[i].sim_gwas(
            params["num_samples_gwas"],
            params["herit_gwas"],
            causal_override=causal_config
        )

    qtl_data_p1 = [None, None]
    gwas_data_p1 = [None, None]
    for i in range(2):
        qtl_data_p2[i] = loci_p2[i].sim_asqtl(
            params["num_samples_qtl"],
            params["coverage"],
            params["std_al_dev"],
            params["herit_qtl"],
            params["herit_as"],
            params["overdispersion"],
            causal_override=causal_config
        )
        gwas_data_p2[i] = loci_p2[i].sim_gwas(
            params["num_samples_gwas"],
            params["herit_gwas"],
            causal_override=causal_config
        )

    return_vals = [
        (loci_p1[0], qtl_data_p1[0], gwas_data_p1[0], causal_config, causal_config,),
        (loci_p1[1], qtl_data_p1[1], gwas_data_p1[1], causal_config, causal_config,),
        (loci_p2[0], qtl_data_p2[0], gwas_data_p2[0], causal_config, causal_config,),
        (loci_p2[1], qtl_data_p2[1], gwas_data_p2[1], causal_config, causal_config,),
    ]

    return return_vals

def sim_unshared_causal(vcf_dir, vcf_name_template, sample_filters, snp_filter, params):
    while True:
        try:
            chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)
            # gc.collect()
            with time_limit(100):
                locus = LocusSimulator(
                    vcf_path, 
                    chrom_num, 
                    start, 
                    params["num_causal"],
                    region_size=params["region_size"],
                    max_snps=params["max_snps"],
                    sample_filter=sample_filters[0],
                    snp_filter=snp_filter,
                    maf_thresh=params["maf_thresh"]
                )
        except (ValueError, TimeoutException):
            continue
        if locus.snp_count >= 10:
            break

    causal_inds_combined = np.random.choice(
        params["num_causal"]*2, 
        locus.snp_count, 
        replace=False
    )
    causal_inds_qtl = causal_inds_combined[:params["num_causal"]]
    causal_inds_gwas = causal_inds_combined[params["num_causal"]:]
    causal_config_qtl = np.zeros(locus.snp_count)
    np.put(causal_config_qtl, causal_inds, 1)
    causal_config_gwas = np.zeros(locus.snp_count)
    np.put(causal_config_gwas, causal_inds, 1)

    qtl_data = locus.sim_asqtl(
        params["num_samples_qtl"],
        params["coverage"],
        params["std_al_dev"],
        params["herit_qtl"],
        params["herit_as"],
        params["overdispersion"],
        causal_override=causal_config_qtl
    )

    gwas_data = locus.sim_gwas(
        params["num_samples_gwas"],
        params["herit_gwas"],
        causal_override=causal_config_gwas
    )

    return locus, qtl_data, gwas_data, causal_config_qtl, causal_config_gwas

def sim_unshared_corr(vcf_dir, vcf_name_template, sample_filters, snp_filter, params):
    max_corr = -1
    while max_corr < params["corr_thresh"]:
        chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)
        try:
            # gc.collect()
            with time_limit(100):
                locus = LocusSimulator(
                    vcf_path, 
                    chrom_num, 
                    start, 
                    1,
                    region_size=params["region_size"],
                    max_snps=params["max_snps"],
                    sample_filter=sample_filters[0],
                    snp_filter=snp_filter,
                    maf_thresh=params["maf_thresh"]
                )
        except (ValueError, TimeoutException):
            continue
        if locus.snp_count < 10:
            continue

        covdiag = np.diag(locus.haps_cov)
        corr = locus.haps_cov / np.sqrt(np.outer(covdiag, covdiag))
        # print(corr) ####
        np.fill_diagonal(corr, -1.) 

        max_idx = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
        max_corr = corr[max_idx]

    valid_pairs = np.greater_equal(corr, params["corr_thresh"])
    valid_idx = np.nonzero(valid_pairs.astype(int))
    # print(valid_idx) ####
    # print(np.shape(valid_idx[0])) ####
    choice_num = np.random.choice(np.arange(np.shape(valid_idx[0])[0]))
    choice_idx = np.stack(valid_idx, axis=-1)[choice_num]

    causal_pair = np.random.permutation(choice_idx)
    
    causal_config_qtl = np.zeros(locus.snp_count)
    np.put(causal_config_qtl, causal_pair[0], 1)
    # print(causal_config_qtl) ####
    causal_config_gwas = np.zeros(locus.snp_count)
    np.put(causal_config_gwas, causal_pair[1], 1)

    qtl_data = locus.sim_asqtl(
        params["num_samples_qtl"],
        params["coverage"],
        params["std_al_dev"],
        params["herit_qtl"],
        params["herit_as"],
        params["overdispersion"],
        causal_override=causal_config_qtl
    )

    gwas_data = locus.sim_gwas(
        params["num_samples_gwas"],
        params["herit_gwas"],
        causal_override=causal_config_gwas
    )

    return locus, qtl_data, gwas_data, causal_config_qtl, causal_config_gwas

def run_model(inputs, model_name, model_qtl_updates):
    if inputs["num_samples_qtl"] == "Perfect":
        causal_set_qtl = inputs["causal_config_qtl"]
        ppas_qtl = inputs["causal_config_qtl"]
        model_qtl = DummyFinemap(ppas_qtl, causal_set_qtl)
    else:
        inputs_qtl = inputs.copy()
        qtl_updates = {
            "total_exp_herit_prior": inputs_qtl["herit_qtl"],
            "imbalance_herit_prior": inputs_qtl["herit_as"],
            "num_ppl": inputs_qtl["num_samples_qtl"],
        }
        inputs_qtl.update(qtl_updates)
        inputs_qtl.update(model_qtl_updates)

        model_qtl = Finemap(**inputs_qtl)
        model_qtl.initialize()

        if inputs_qtl["search_mode"] == "exhaustive":
            model_qtl.search_exhaustive(inputs["min_causal"], inputs["max_causal"])
        elif inputs_qtl["search_mode"] == "shotgun":
            model_qtl.search_shotgun(
                inputs_qtl["min_causal"], 
                inputs_qtl["max_causal"], 
                inputs_qtl["prob_threshold"], 
                inputs_qtl["streak_threshold"], 
                inputs_qtl["search_iterations"]
            )

        causal_set_qtl = model_qtl.get_causal_set(inputs_qtl["confidence"])
        ppas_qtl = model_qtl.get_ppas()

    if inputs["num_samples_gwas"] == "Perfect":
        causal_set_gwas = inputs["causal_config_gwas"]
        ppas_gwas = inputs["causal_config_gwas"]
        model_gwas = DummyFinemap(ppas_gwas, causal_set_gwas)
    else:
        inputs_gwas = inputs.copy()
        gwas_updates = {
            "total_exp_herit_prior": inputs_gwas["herit_gwas"],
            "total_exp_stats": inputs_gwas["z_gwas"],
            "total_exp_corr": inputs_gwas["ld_gwas"],
            "num_ppl": inputs_gwas["num_samples_gwas"],
            "qtl_only": True
        }
        inputs_gwas.update(gwas_updates)

        model_gwas = Finemap(**inputs_gwas)
        model_gwas.initialize()

        if inputs_gwas["search_mode"] == "exhaustive":
            model_gwas.search_exhaustive(inputs["min_causal"], inputs["max_causal"])
        elif inputs_gwas["search_mode"] == "shotgun":
            model_gwas.search_shotgun(
                inputs_gwas["min_causal"], 
                inputs_gwas["max_causal"], 
                inputs_gwas["prob_threshold"], 
                inputs_gwas["streak_threshold"], 
                inputs_gwas["search_iterations"]
            )

        causal_set_gwas = model_gwas.get_causal_set(inputs_qtl["confidence"])
        ppas_gwas = model_gwas.get_ppas()

    clpps = model_qtl.coloc_clpps(model_gwas)
    h0, h1, h2, h3, h4 = model_qtl.coloc_hyps(model_gwas)

    # print(model_name) ####
    # print(model_qtl.imbalance_stats) ####
    # print(model_qtl.total_exp_stats) ####
    # print(ppas_qtl) ####
    # print(inputs_gwas["z_gwas"]) ####
    # print(ppas_gwas) ####
    # print(clpps) ####
    # print(h4) ####
    # if model_name == "eqtl": ####
    #   print(model_qtl.beta) ####
    #   print(model_qtl.total_exp_stats) ####
    #   print(model_qtl.total_exp_stats[inputs["causal_config_qtl"].astype(bool)])
    #   print(ppas_qtl[inputs["causal_config_qtl"].astype(bool)]) ####
    #   print(ppas_gwas) ####
    #   print(ppas_gwas[inputs["causal_config_qtl"].astype(bool)])
        # print(np.sum(ppas_qtl)) ####
        # print(np.corrcoef(inputs_gwas["z_gwas"], model_qtl.total_exp_stats)) ####
        # print(clpps) ####
        # print(h4) ####

    result = {
        "causal_set_qtl": causal_set_qtl,
        "causal_set_gwas": causal_set_gwas,
        "ppas_qtl": ppas_qtl,
        "ppas_gwas": ppas_gwas,
        "clpps": clpps,
        "h0": h0,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4
    }

    result.update(inputs)
    result["model"] = model_name
    result["complete"] = True
    return result

def run_ecav(inputs, model_name, model_qtl_updates):
    inputs_qtl = inputs.copy()
    qtl_updates = {
        "total_exp_herit_prior": inputs_qtl["herit_qtl"],
        "imbalance_herit_prior": inputs_qtl["herit_as"],
        "num_ppl": inputs_qtl["num_samples_qtl"],
    }
    inputs_qtl.update(qtl_updates)
    inputs_qtl.update(model_qtl_updates)

    model_qtl = Finemap(**inputs_qtl)
    model_qtl.initialize()

    inputs_gwas = inputs.copy()
    gwas_updates = {
        "total_exp_herit_prior": inputs_gwas["herit_gwas"],
        "total_exp_stats": inputs_gwas["z_gwas"],
        "total_exp_corr": inputs_gwas["ld_gwas"],
        "num_ppl": inputs_gwas["num_samples_gwas"],
        "qtl_only": True
    }
    inputs_gwas.update(gwas_updates)

    model_gwas = Finemap(**inputs_gwas)
    model_gwas.initialize()

    model_ecaviar = EvalECaviar(
        model_qtl,
        model_gwas, 
        inputs["confidence"], 
        inputs["max_causal"]
    )
    model_ecaviar.run()

    causal_set_qtl = model_ecaviar.causal_set_qtl
    causal_set_gwas = model_ecaviar.causal_set_gwas
    ppas_qtl = model_ecaviar.post_probs_qtl
    ppas_gwas = model_ecaviar.post_probs_gwas

    if (inputs["num_samples_qtl"] == "Perfect") and (inputs["num_samples_gwas"] == "Perfect"):
        clpps = inputs["causal_config_qtl"] * inputs["causal_config_gwas"]
        h4 = np.sum(clpps)
    elif inputs["num_samples_qtl"] == "Perfect":
        clpps = inputs["causal_config_qtl"] * ppas_gwas
        h4 = np.sum(clpps)
    elif inputs["num_samples_gwas"] == "Perfect":
        clpps = inputs["causal_config_gwas"] * ppas_qtl
        h4 = np.sum(clpps)
    else:
        clpps = model_ecaviar.clpp
        h4 = model_ecaviar.h4

    # print("ecav") ####
    # print(ppas_qtl) ####
    # print(np.sum(ppas_qtl)) ####
    # print(ppas_gwas) ####
    # print(np.sum(ppas_gwas)) ####
    # print(clpps) ####
    # print(h4) ####

    result = {
        "causal_set_qtl": causal_set_qtl,
        "causal_set_gwas": causal_set_gwas,
        "ppas_qtl": ppas_qtl,
        "ppas_gwas": ppas_gwas,
        "clpps": clpps,
        "h4": h4
    }

    result.update(inputs)
    result["model"] = model_name
    result["complete"] = True
    return result

def coloc_test(
        out_dir, 
        batch_size, 
        batch_num, 
        params_path
    ):
    with open(params_path, "rb") as params_file:
        params = pickle.load(params_file)
    test_type = params["test_type"]
    vcf_dir = params["vcf_dir"] 
    vcf_name_template = params["vcf_name_template"]
    model_flavors = params["model_flavors"]

    sample_filter_data = pd.read_csv(
        params["sample_filter_path"], 
        sep="\t", 
        usecols=["sample", "super_pop"]
    )
    
    sample_filter_eur = set(
        sample_filter_data.loc[
            sample_filter_data["super_pop"]=="EUR",
            ["sample"]
        ].to_numpy().flatten()
    )

    sample_filter_afr = set(
        sample_filter_data.loc[
            sample_filter_data["super_pop"]=="AFR",
            ["sample"]
        ].to_numpy().flatten()
    )

    sample_filters = [sample_filter_eur, sample_filter_afr]

    with open(params["snp_filter_path"], "rb") as snp_filter_file:
        snp_filter = pickle.load(snp_filter_file)
    
    

    qtl_updates_map = {
        "full": {},
        "indep": {"cross_corr_prior": 0.},
        "ase": {"as_only": True},
        "ecav": {"qtl_only": True},
        "eqtl": {"qtl_only": True}
    }

    sim_map = {
        "shared": sim_shared_causal,
        # "unshared": sim_unshared_causal,
        "corr": sim_unshared_corr,
        "shared_xpop": sim_shared_causal_xpop,
    }
    multi_out_fns = set(["shared_xpop"])

    sim_fn = sim_map[test_type]
    multi_out = (test_type in multi_out_fns)

    output = []
    for _ in range(batch_size):
        try:
            fn_out = sim_fn(
                vcf_dir, 
                vcf_name_template, 
                sample_filters,
                snp_filter,
                params
            )
            for ind, l in enumerate([fn_out] if not multi_out else fn_out):
                locus, qtl_data, gwas_data, causal_config_qtl, causal_config_gwas = l
                # print(qtl_data) ####
                # print(qtl_data["counts_A"]) ####
                # print(qtl_data["counts_B"]) ####
                sim_data = {
                    "sim_type": test_type,
                    "chrom": locus.chrom,
                    "locus_pos": locus.start,
                    "locus_size": locus.region_size,
                    "snp_ids": locus.snp_ids,
                    "num_snps": locus.snp_count,
                    "causal_config_qtl": causal_config_qtl,
                    "causal_config_gwas": causal_config_gwas
                }
                sim_data.update(qtl_data)
                sim_data.update(gwas_data)

                inputs = params.copy()
                inputs.update(sim_data)
                # print(inputs["hap_A"].shape()) ####

                for m in model_flavors:
                    try:
                        model_qtl_updates = qtl_updates_map[m]
                        if m == "ecav":
                            result = run_ecav(inputs, "ecav", model_qtl_updates)
                        else:
                            result = run_model(inputs, m, model_qtl_updates)
                        result["res_set"] = ind
                    except Exception as e:
                        # raise ####
                        trace = traceback.format_exc()
                        print(trace) ####
                        message = repr(e)
                        result = {
                            "model": m,
                            "complete": False, 
                            "error": message, 
                            "traceback": trace,
                            "res_set": ind,
                        }
                    finally:
                        output.append(result)

        except Exception as e:
            # raise ####
            trace = traceback.format_exc()
            print(trace) ####
            message = repr(e)
            result = {"complete": False, "error": message, "traceback": trace, "res_set": None}
            for m in model_flavors:
                result_model = result.copy()
                result_model.update({"model": m})
                output.append(result_model)

            continue

        
            
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output_name = "{0}_out_{1}.pickle".format(params["test_name"], batch_num)
    output_return = os.path.join(out_dir, output_name)
    with open(output_return, "wb") as output_file:
        pickle.dump(output, output_file)

if __name__ == '__main__':
    out_dir = sys.argv[1]
    batch_size = int(sys.argv[2])
    batch_num = int(sys.argv[3])
    params_path = sys.argv[4]
    coloc_test(
        out_dir, 
        batch_size, 
        batch_num, 
        params_path
    )