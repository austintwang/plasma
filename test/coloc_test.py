#!/usr/bin/env python3

import os
import random
import traceback
import pickle

import numpy as np
import pandas as pd

if __name__ == '__main__' and __package__ is None:
	__package__ = 'test'
	import sys
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import Finemap, LocusSimulator, EvalECaviar

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
	regions = [i - 1000000 for i in CHROM_LENS]
	weights = np.array(regions) / np.sum(regions)
	chrom_num = np.random.choice(list(range(1, 23)), p=weights)
	chrom = "chr{0}".format(chrom_num)
	vcf_path = os.path.join(vcf_dir, vcf_name_template.format(chrom))
	start = random.randrange(0, regions[chrom_num-1])

	return chrom, chrom_num, start, vcf_path

def sim_shared_causal(vcf_dir, vcf_name_template, sample_filter, snp_filter, params):
	chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)

	while True:
		locus = LocusSimulator(
			vcf_path, 
			chrom_num, 
			start, 
			params["num_causal"],
			region_size=params["region_size"],
			max_snps=params["max_snps"],
			sample_filter=sample_filter,
			snp_filter=snp_filter,
			maf_thresh=params["maf_thresh"]
		)
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

	gwas_data = locus.sim_asqtl(
		params["num_samples_gwas"],
		params["herit_gwas"],
	)

	return locus, qtl_data, gwas_data, causal_config_qtl, causal_config_gwas

def sim_unshared_causal(vcf_dir, vcf_name_template, sample_filter, snp_filter, params):
	chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)

	while True:
		locus = LocusSimulator(
			vcf_path, 
			chrom_num, 
			start, 
			params["num_causal"],
			region_size=params["region_size"],
			max_snps=params["max_snps"],
			sample_filter=sample_filter,
			snp_filter=snp_filter,
			maf_thresh=params["maf_thresh"]
		)
		if locus.snp_count >= 10:
			break

	causal_inds_combined = np.random.choice(
		params["num_causal"]*2, 
		params["num_snps"], 
		replace=False
	)
	causal_inds_qtl = causal_inds_combined[:params["num_causal"]]
	causal_inds_gwas = causal_inds_combined[params["num_causal"]:]
	causal_config_qtl = np.zeros(params["num_snps"])
	np.put(causal_config_qtl, causal_inds, 1)
	causal_config_gwas = np.zeros(params["num_snps"])
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

	gwas_data = locus.sim_asqtl(
		params["num_samples_gwas"],
		params["herit_gwas"],
		causal_override=causal_config_gwas
	)

	return locus, qtl_data, gwas_data, causal_config_qtl, causal_config_gwas

def sim_unshared_corr(vcf_dir, vcf_name_template, sample_filter, snp_filter, params):
	max_corr = 0.
	while max_corr < params["corr_thresh"]:
		chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)

		locus = LocusSimulator(
			vcf_path, 
			chrom_num, 
			start, 
			1,
			region_size=params["region_size"],
			max_snps=params["max_snps"],
			sample_filter=sample_filter,
			snp_filter=snp_filter,
			maf_thresh=params["maf_thresh"]
		)
		if locus.snp_count < 10:
			continue

		covdiag = np.diag(locus.haps_cov)
		corr = locus.haps_cov / np.sqrt(np.outer(covdiag, covdiag))
		np.fill_diagonal(corr, 0.) 

		max_idx = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
		max_corr = corr[max_idx]

	valid_pairs = np.greater_equal(corr, params["corr_thresh"])
	valid_idx = np.nonzero(valid_pairs.astype(int))
	choice_idx = np.random.choice(np.stack(valid_idx, axis=-1))

	causal_pair = np.random.permutation(choice_idx)
	
	causal_config_qtl = np.zeros(params["num_snps"])
	np.put(causal_config_qtl, causal_pair[0], 1)
	causal_config_gwas = np.zeros(params["num_snps"])
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

	gwas_data = locus.sim_asqtl(
		params["num_samples_gwas"],
		params["herit_gwas"],
		causal_override=causal_config_gwas
	)

	return locus, qtl_data, gwas_data, causal_config_qtl, causal_config_gwas

def run_model(inputs, model_name, model_qtl_updates):
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

	inputs_gwas = inputs.copy()
	gwas_updates = {
		"total_exp_herit_prior": inputs_gwas["herit_gwas"],
		"total_exp_stats": inputs_gwas["z_gwas"],
		"total_exp_corr": inputs_gwas["ld_gwas"],
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

	causal_set_gwas = model_qtl.get_causal_set(inputs_qtl["confidence"])
	ppas_gwas = model_qtl.get_ppas()

	clpps = model_qtl.coloc_clpps(model_gwas)
	h0, h1, h2, h3, h4 = model_qtl.coloc_hyps(model_gwas)

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
	clpps = model_ecaviar.clpp
	h4 = model_ecaviar.h4

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
	
	sample_filter = set(
		sample_filter_data.loc[
			sample_filter_data["super_pop"]=="EUR",
			["sample"]
		].to_numpy().flatten()
	)

	with open(params["snp_filter_path"], "rb") as snp_filter_file:
		snp_filter = pickle.load(snp_filter_file)
	
	sim_map = {
		"shared": sim_shared_causal,
		"unshared": sim_shared_causal,
		"corr": sim_unshared_corr
	}
	sim_fn = sim_map[test_type]

	output = []
	for _ in range(batch_size):
		try:
			locus, qtl_data, gwas_data, causal_config_qtl, causal_config_gwas = sim_fn(
				vcf_dir, 
				vcf_name_template, 
				sample_filter,
				snp_filter,
				params
			)
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

		except Exception as e:
			raise ####
			trace = traceback.format_exc()
			message = repr(e)
			result = {"complete": False, "error": message, "traceback": trace}
			if "full" in model_flavors:
				result_full = result.copy()
				result_full.update({"model": "full"})
				output.append(result_full)
			if "indep" in model_flavors:
				result_indep = result.copy()
				result_indep.update({"model": "indep"})
				output.append(result_indep)
			if "ase" in model_flavors:
				result_ase = result.copy()
				result_ase.update({"model": "ase"})
				output.append(result_ase)
			if "ecav" in model_flavors:
				result_ecav = result.copy()
				result_ecav.update({"model": "ecav"})
				output.append(result_ecav)
			if "eqtl" in model_flavors:
				result_eqtl = result.copy()
				result_eqtl.update({"model": "eqtl"})
				output.append(result_eqtl)

			continue

		if "full" in model_flavors:
			try:
				model_qtl_updates = {}
				result_full = run_model(inputs, "full", model_qtl_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_full = {
					"model": "full",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_full)

		if "indep" in model_flavors:
			try:
				model_qtl_updates = {"cross_corr_prior": 0.}
				result_indep = run_model(inputs, "indep", model_qtl_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_indep = {
					"model": "indep",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_indep)
			
		if "ase" in model_flavors:
			try:
				model_qtl_updates = {"as_only": True}
				result_ase = run_model(inputs, "ase", model_qtl_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_ase = {
					"model": "ase",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_ase)
			
		if "ecav" in model_flavors:
			try:
				model_qtl_updates = {"qtl_only": True}
				result_ecav = run_ecav(inputs, "ecav", model_qtl_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_ecav = {
					"model": "ecav",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_ecav)
			
		if "eqtl" in model_flavors:
			try:
				model_qtl_updates = {"qtl_only": True}
				result_eqtl = run_model(inputs, "ase", model_qtl_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_eqtl = {
					"model": "eqtl",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_eqtl)
			
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