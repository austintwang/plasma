#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import os
import random
import numpy as np
import pandas as pd

from . import Finemap, LocusSimulator
from . import EvalCaviar, EvalCaviarASE

VCF_NAME_TEMPLATE = "ALL.{0}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
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

def draw_region(vcf_dir, vcf_name_template):
	regions = [i - 500000 for i in CHROM_LENS]
	weights = np.array(regions) / np.sum(regions)
	chrom_num = np.random.choice(range(1, 23), p=weights)
	chrom = "chr{0}".format(chrom_num)
	vcf_path = os.path.join(vcf_dir, vcf_name_template.format(chrom))
	start = random.randrange(0, regions[chrom_num-1])

	return chrom, start, vcf_path

def sim_shared_causal(vcf_dir, vcf_name_template, pop_name, shared_params, qtl_params, gwas_params):
	pop_data = pd.read_csv("pop_name", sep="\t", usecols=["sample", "super_pop"])
	pop_fiter = pop_data.loc[pop_data["super_pop"] == "EUR", ["sample"]].to_numpy().flatten() 

	chrom, start, vcf_path = draw_region(vcf_dir)

	locus = LocusSimulator(
		vcf_path, 
		chrom, 
		start, 
		shared_params["num_snps"], 
		shared_params["num_causal"],
		sample_filter=pop_fiter,
		maf_thresh=0.1
	)

	qtl_data = locus.sim_asqtl(
		qtl_params["num_samples"]
		qtl_params["coverage"],
		qtl_params["std_al_dev"],
		qtl_params["herit_qtl"],
		qtl_params["herit_as"],
		qtl_params["overdispersion"],
	)

	gwas_z = locus.sim_asqtl(
		gwas_params["num_samples"]
		gwas_params["herit"],
	)

	return locus, qtl_data, gwas_z

def sim_unshared_causal(vcf_dir, vcf_name_template, pop_name, shared_params, qtl_params, gwas_params):
	pop_data = pd.read_csv("pop_name", sep="\t", usecols=["sample", "super_pop"])
	pop_fiter = pop_data.loc[pop_data["super_pop"] == "EUR", ["sample"]].to_numpy().flatten() 

	chrom, start, vcf_path = draw_region(vcf_dir)

	locus = LocusSimulator(
		vcf_path, 
		chrom, 
		start, 
		shared_params["num_snps"], 
		shared_params["num_causal"],
		sample_filter=pop_fiter,
		maf_thresh=shared_params["maf_thresh"]
	)

	causal_inds_combined = np.random.choice(
		shared_params["num_causal"]*2, 
		shared_params["num_snps"], 
		replace=False
	)
	causal_inds_qtl = causal_inds_combined[:shared_params["num_causal"]]
	causal_inds_gwas = causal_inds_combined[shared_params["num_causal"]:]
	causal_config_qtl = np.zeros(shared_params["num_snps"])
	np.put(causal_config_qtl, causal_inds, 1)
	causal_config_gwas = np.zeros(shared_params["num_snps"])
	np.put(causal_config_gwas, causal_inds, 1)

	qtl_data = locus.sim_asqtl(
		qtl_params["num_samples"]
		qtl_params["coverage"],
		qtl_params["std_al_dev"],
		qtl_params["herit_qtl"],
		qtl_params["herit_as"],
		qtl_params["overdispersion"],
		causal_override=causal_config_qtl
	)

	gwas_z = locus.sim_asqtl(
		gwas_params["num_samples"]
		gwas_params["herit"],
		causal_override=causal_config_gwas
	)

	return locus, qtl_data, gwas_z

def sim_unshared_corr(vcf_dir, vcf_name_template, pop_name, shared_params, qtl_params, gwas_params):
	pop_data = pd.read_csv("pop_name", sep="\t", usecols=["sample", "super_pop"])
	pop_fiter = pop_data.loc[pop_data["super_pop"] == "EUR", ["sample"]].to_numpy().flatten() 

	max_corr = 0.
	while max_corr < shared_params["corr_thresh"]:
		chrom, start, vcf_path = draw_region(vcf_dir)

		locus = LocusSimulator(
			vcf_path, 
			chrom, 
			start, 
			shared_params["num_snps"], 
			1,
			sample_filter=pop_fiter,
			maf_thresh=shared_params["maf_thresh"]
		)

		covdiag = np.diag(locus.haps_cov)
		corr = locus.haps_cov / np.sqrt(np.outer(covdiag, covdiag))
		np.fill_diagonal(corr, 0.) 
		max_idx = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
		max_corr = corr[max_idx]

	causal_pair = np.random.permutation(max_idx)
	
	causal_config_qtl = np.zeros(shared_params["num_snps"])
	np.put(causal_config_qtl, causal_pair[0], 1)
	causal_config_gwas = np.zeros(shared_params["num_snps"])
	np.put(causal_config_gwas, causal_pair[1], 1)

	qtl_data = locus.sim_asqtl(
		qtl_params["num_samples"]
		qtl_params["coverage"],
		qtl_params["std_al_dev"],
		qtl_params["herit_qtl"],
		qtl_params["herit_as"],
		qtl_params["overdispersion"],
		causal_override=causal_config_qtl
	)

	gwas_z = locus.sim_asqtl(
		gwas_params["num_samples"]
		gwas_params["herit"],
		causal_override=causal_config_gwas
	)

	return locus, qtl_data, gwas_z



def coloc_test(test_type, vcf_dir, vcf_name_template, out_dir, batch_size, shared_params, qtl_params, gwas_params):


