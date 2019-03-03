#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

# print("imp1") ####

import numpy as np
import os
import sys
import traceback

# print("hi", file=sys.stderr) ####
# print("imp2") ####

try:
	import cPickle as pickle
except ImportError:
	import pickle

# print("imp3") ####

# print(__package__) ####
if __name__ == '__main__' and __package__ is None:
	__package__ = 'run'
	from eval_caviar import EvalCaviarASE
else:
	from .eval_caviar import EvalCaviarASE

# print("imp4") ####

# print(__package__) ####

# import eval_caviar
# from .eval_caviar import EvalCaviarASE
# from . import eval_caviar
# import run.eval_caviar.EvalCaviarASE as EvalCaviarASE

try:
	import Finemap
except ImportError:
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
	from ase_finemap import Finemap

# print("imp5") ####

def run_model(inputs, input_updates, informative_snps):
	model_inputs = inputs.copy()
	model_inputs.update(input_updates)
	# print(model_inputs) ####

	model = Finemap(**model_inputs)
	model.initialize()

	if inputs["search_mode"] == "exhaustive":
		model.search_exhaustive(inputs["min_causal"], inputs["max_causal"])
	elif inputs["search_mode"] == "shotgun":
		model.search_shotgun(
			inputs["min_causal"], 
			inputs["max_causal"], 
			inputs["prob_threshold"], 
			inputs["streak_threshold"], 
			inputs["search_iterations"]
		)

	causal_set_inf = model.get_causal_set(inputs["confidence"])
	ppas_inf = model.get_ppas()
	size_probs = model.get_size_probs()

	causal_set = np.ones(np.shape(inputs["snp_ids"]))
	np.put(causal_set, informative_snps, causal_set_inf)

	ppas = np.full(np.shape(inputs["snp_ids"]), np.nan)
	np.put(ppas, informative_snps, ppas_inf)

	return causal_set, ppas, size_probs, model

def get_ldsr_data(inputs, causal_set, ppas):
	cset_bool = (np.array(causal_set) == 1)

	chromosome = inputs["chr"]
	markers = inputs["snp_ids"][cset_bool]
	positions = inputs["snp_pos"][cset_bool]
	ends = positions + 1
	ppas_cset = np.array(ppas)[cset_bool]
	gene = inputs["name"]

	data = {}
	for ind, val in enumerate(markers):
		data[val] = {
			"chr": chromosome,
			"start": positions[ind], 
			"end": ends[ind], 
			"ppa": ppas_cset[ind], 
			"gene": gene
		}	

	return data

def write_output(output_path, result):
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	output_return = os.path.join(output_path, "output.pickle")
	with open(output_return, "wb") as output_file:
		pickle.dump(result, output_file)

def main(output_path, input_path, params_path, selection_path, filter_path):
	# input_path = os.path.join(target_dir, "input.pickle")
	if selection_path == "all":
		selection = False
	else:
		with open(selection_path, "rb") as selection_file:
			selection = pickle.load(selection_file)

	if filter_path == "all":
		snp_filter = False
	else:
		with open(filter_path, "rb") as filter_file:
			snp_filter = pickle.load(filter_file)

	with open(input_path, "rb") as input_file:
		# print(input_path) ####
		inputs = pickle.load(input_file)

	with open(params_path, "rb") as params_file:
		# print(input_path) ####
		params = pickle.load(params_file)

	inputs.update(params)

	if selection:
		select = np.array([i in selection for i in inputs["sample_names"]])

		# num_ppl_raw = np.size(inputs["counts1"])
		# max_ppl = hyperparams.get("max_ppl")
		# if max_ppl and max_ppl < num_ppl_raw:
		# 	threshold = np.array([1] * max_ppl + [0] * (num_ppl_raw - max_ppl))
		# 	np.random.shuffle(threshold)
		# 	select = np.logical_and(select, threshold)
		# 	inputs["num_ppl"] = max_ppl

		inputs["hap1"] = inputs["hap1"][select]
		inputs["hap2"] = inputs["hap2"][select]
		inputs["counts1"] = inputs["counts1"][select]
		inputs["counts2"] = inputs["counts2"][select]
		inputs["counts_total"] = inputs["counts_total"][select]

	num_ppl_raw = np.size(inputs["counts1"])

	max_ppl = inputs.get("max_ppl")
	if max_ppl and max_ppl < num_ppl_raw:
		threshold = np.array([1] * max_ppl + [0] * (num_ppl_raw - max_ppl)).astype(np.bool)
		# print(threshold) ####
		np.random.shuffle(threshold)
		# print(threshold) ####
		# print(np.size(inputs["counts1"])) ####
		inputs["hap1"] = inputs["hap1"][threshold]
		inputs["hap2"] = inputs["hap2"][threshold]
		inputs["counts1"] = inputs["counts1"][threshold]
		inputs["counts2"] = inputs["counts2"][threshold]
		inputs["counts_total"] = inputs["counts_total"][threshold]
		# print(np.size(inputs["counts1"])) ####

	select_counts = np.logical_and(inputs["counts1"] >= 1, inputs["counts2"] >= 1) 
	
	inputs["hap1"] = inputs["hap1"][select_counts]
	inputs["hap2"] = inputs["hap2"][select_counts]
	inputs["counts1"] = inputs["counts1"][select_counts]
	inputs["counts2"] = inputs["counts2"][select_counts]
	inputs["counts_total"] = inputs["counts_total"][select_counts]

	inputs["num_ppl"] = np.size(inputs["counts1"])

	if snp_filter:
		snps_in_filter = [ind for ind, val in enumerate(inputs["snp_ids"]) if val in snp_filter]
		inputs["snp_ids"] = inputs["snp_ids"][snps_in_filter]
		inputs["snp_pos"] = inputs["snp_pos"][snps_in_filter]
		inputs["hap1"] = inputs["hap1"][:, snps_in_filter]
		inputs["hap2"] = inputs["hap2"][:, snps_in_filter]

	# inputs["num_snps_imbalance"] = len(inputs["hap1"])
	# inputs["num_snps_total_exp"] = inputs["num_snps_imbalance"]

	haps_comb = inputs["hap1"] + inputs["hap2"]
	informative_snps = np.where(np.all(haps_comb == haps_comb[0,:], axis=0))[0]
	# print(informative_snps) ####

	# inputs["snp_ids"] = inputs["snp_ids"][informative_snps]
	# inputs["snp_pos"] = inputs["snp_pos"][informative_snps]
	inputs["hap1"] = inputs["hap1"][:, informative_snps]
	inputs["hap2"] = inputs["hap2"][:, informative_snps]

	inputs["num_snps_imbalance"] = len(inputs["hap1"])
	inputs["num_snps_total_exp"] = inputs["num_snps_imbalance"]

	result = {}

	if inputs["num_ppl"] == 0:
		result["data_error"] = "Insufficient Read Counts"
		write_output(output_path, result)
		return

	if inputs["hap1"].size == 0:
		result["data_error"] = "Insufficient Markers"
		write_output(output_path, result)
		return

	# print(inputs["num_ppl"]) ####
	# print(max_ppl) ####

	num_ppl = inputs["num_ppl"]
	num_causal = inputs["num_causal"]
	eqtl_herit = 1 - inputs["prop_noise_eqtl"]
	ase_herit = 1 - inputs["prop_noise_ase"]

	coverage = np.mean(inputs["counts1"] + inputs["counts2"])
	overdispersion = inputs["overdispersion"]
	# std_fraction = inputs["std_fraction"]
	# ase_inherent_var = (np.log(std_fraction) - np.log(1-std_fraction))**2
	imbalance = np.log(inputs["counts1"]) - np.log(inputs["counts2"])
	ase_inherent_var = np.var(imbalance)
	counts = np.mean(inputs["counts1"] + inputs["counts2"])
	# print(ase_inherent_var) ####

	ase_count_var = (
		2 / coverage
		* (
			1 
			+ (
				1
				/ (
					1 / (np.exp(ase_inherent_var / 2))
					+ 1 / (np.exp(ase_inherent_var / 2)**3)
					* (
						(np.exp(ase_inherent_var * 2) + 1) / 2
						- np.exp(ase_inherent_var)
					)
				)
			)
		)
		* (1 + overdispersion * (coverage - 1))
	)
	correction = ase_inherent_var / (ase_inherent_var + ase_count_var)
	ase_herit_adj = ase_herit * correction

	corr_stats = np.sqrt(
		(num_ppl / num_causal)**2 * eqtl_herit * ase_herit_adj
		/ (
			(1 + eqtl_herit * (num_ppl / num_causal - 1))
			* (1 + ase_herit_adj * (num_ppl / num_causal - 1))
		)
	)

	iv = (
		(num_ppl / num_causal * ase_herit_adj / (1 - ase_herit_adj)) 
	)
	xv = (
		(num_ppl / num_causal * eqtl_herit / (1 - eqtl_herit)) 
	)

	imbalance_var_prior = 1 * iv
	total_exp_var_prior = 1 * xv

	inputs["hap_A"] = inputs["hap1"].astype(np.int)
	inputs["hap_B"] = inputs["hap2"].astype(np.int)

	inputs["counts_A"] = inputs["counts1"].astype(np.int)
	inputs["counts_B"] = inputs["counts2"].astype(np.int)
	inputs["total_exp"] = inputs["counts_total"].astype(np.int)

	inputs["num_ppl_imbalance"] = num_ppl
	inputs["num_ppl_total_exp"] = num_ppl

	inputs["num_snps_imbalance"] = np.shape(inputs["hap_A"])[1]
	inputs["num_snps_total_exp"] = inputs["num_snps_imbalance"]

	inputs["causal_status_prior"] = 1 / inputs["num_snps_imbalance"]

	# print(np.shape(inputs["hap_A"])) ####
	# print(np.shape(inputs["hap_B"])) ####
	# print(np.shape(inputs["counts_A"])) ####
	# print(np.shape(inputs["counts_B"])) ####
	# print(num_ppl) ####
	# raise Exception ####

	if inputs["model_flavors"] == "all":
		model_flavors = set(["full", "indep", "eqtl", "ase", "acav"])
	else:
		model_flavors = inputs["model_flavors"]

	updates_full = {
		"corr_stats": corr_stats,
		"imbalance_var_prior": imbalance_var_prior,
		"total_exp_var_prior": total_exp_var_prior
	}
	if "full" in model_flavors:
		result["causal_set_full"], result["ppas_full"], result["size_probs_full"], model_full = run_model(
			inputs, updates_full,informative_snps
		)
		result["ldsr_data_full"] = get_ldsr_data(inputs, result["causal_set_full"], result["ppas_full"])

	updates_indep = {
		"cross_corr_prior": 0.0, 
		"corr_stats": 0.0,
		"imbalance_var_prior": imbalance_var_prior,
		"total_exp_var_prior": total_exp_var_prior
	}
	if "indep" in model_flavors:
		result["causal_set_indep"], result["ppas_indep"], result["size_probs_indep"], model_indep = run_model(
			inputs, updates_indep, informative_snps
		)
		result["ldsr_data_indep"] = get_ldsr_data(inputs, result["causal_set_indep"], result["ppas_indep"])

	updates_eqtl = {
		"counts_A": np.zeros(shape=0),
		"counts_B": np.zeros(shape=0),
		"imbalance": np.zeros(shape=0), 
		"phases": np.zeros(shape=(0,0)),
		"imbalance_corr": np.zeros(shape=(0,0)),
		"imbalance_errors": np.zeros(shape=0),
		"imbalance_stats": np.zeros(shape=0),
		"num_ppl_imbalance": 0,
		"num_snps_imbalance": 0,
		"corr_stats": 0.0,
		"imbalance_var_prior": imbalance_var_prior,
		"total_exp_var_prior": total_exp_var_prior,
		"cross_corr_prior": 0.0,
	}
	if "eqtl" in model_flavors:
		result["causal_set_eqtl"], result["ppas_eqtl"], result["size_probs_eqtl"], model_eqtl = run_model(
			inputs, updates_eqtl, informative_snps
		)
		result["ldsr_data_eqtl"] = get_ldsr_data(inputs, result["causal_set_eqtl"], result["ppas_eqtl"])

	updates_ase = {
		"total_exp": np.zeros(shape=0), 
		"genotypes_comb": np.zeros(shape=(0,0)),
		"total_exp_corr": np.zeros(shape=(0,0)),
		"total_exp_errors": np.zeros(shape=0),
		"total_exp_stats": np.zeros(shape=0),
		"num_ppl_total_exp": 0,
		"num_snps_total_exp": 0,
		"corr_stats": 0.0,
		"imbalance_var_prior": imbalance_var_prior,
		"total_exp_var_prior": total_exp_var_prior,
		"cross_corr_prior": 0.0,
	}
	if "ase" in model_flavors:
		result["causal_set_ase"], result["ppas_ase"], result["size_probs_ase"], model_ase = run_model(
			inputs, updates_ase, informative_snps
		)
		result["ldsr_data_ase"] = get_ldsr_data(inputs, result["causal_set_ase"], result["ppas_ase"])
		# print(result["causal_set_ase"]) ####

	if "acav" in model_flavors:
		model_inputs_dummy = inputs.copy()
		model_inputs_dummy.update(updates_full)
		model_dummy = Finemap(**model_inputs_dummy)
		model_dummy.initialize()
		model_caviar_ase = EvalCaviarASE(
			model_full, 
			inputs["confidence"], 
			inputs["max_causal"]
		)
		model_caviar_ase.run()

		causal_set = np.ones(np.shape(inputs["snp_ids"]))
		np.put(causal_set, informative_snps, model_caviar_ase.causal_set)

		ppas = np.full(np.shape(inputs["snp_ids"]), np.nan)
		np.put(ppas, informative_snps, model_caviar_ase.post_probs)

		result["causal_set_caviar_ase"] = causal_set
		result["ppas_caviar_ase"] = ppas
		result["ldsr_data_caviar_ase"] = get_ldsr_data(
			inputs, result["causal_set_caviar_ase"], result["ppas_caviar_ase"]
		)

	write_output(output_path, result)

	# print(result) ####
	# print(sum(result["causal_set_eqtl"])) ####
	# print(sum(result["causal_set_caviar_ase"])) ####
	# print(sum(result["causal_set_ase"])) ####
	# print(sum(result["causal_set_full"])) ####
	# print(sum(result["causal_set_indep"])) ####

if __name__ == '__main__':
	# data_dir = sys.argv[0]
	# print("woiehofwie") ####
	# __package__ = "run"

	output_path = sys.argv[1]
	input_path = sys.argv[2]
	params_path = sys.argv[3]
	selection_path = sys.argv[4]
	filter_path = sys.argv[5]
	main(output_path, input_path, params_path, selection_path, filter_path)

	
	# exit_code = 0
	# try:
	# 	# print(os.environ) ####
	# 	# data_dir = os.environ["DATA_DIR"]
	# 	output_path = sys.argv[1]
	# 	input_path = sys.argv[2]
	# 	params_path = sys.argv[3]
	# 	main(output_path, input_path, params_path)
	# except Exception:
	# 	print("woiehofwie") ####
	# 	exit_code = 1
	# 	t, v, tb = sys.exc_info()
	# 	raise t, v, tb
	# 	# raise e
	# finally:
	# 	sys.exit(exit_code)
	# 	# pass