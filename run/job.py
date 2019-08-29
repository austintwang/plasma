#!/usr/bin/env python

import numpy as np
import os
import sys
import traceback
import pickle

if __name__ == '__main__' and __package__ is None:
	__package__ = 'run'
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
	sys.path.insert(0, "/agusevlab/awang/plasma")
	
from . import Finemap, Caviar, CaviarASE, FmBenner

# print("imp5") ####

def run_model(model_cls, inputs, input_updates, informative_snps):
	model_inputs = inputs.copy()
	model_inputs.update(input_updates)
	# print(model_inputs) ####

	model = model_cls(**model_inputs)
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

def get_bed_ctrl(inputs):
	chromosome = inputs["chr"]
	markers = inputs["snp_ids"]
	positions = inputs["snp_pos"]
	ends = positions + 1
	gene = inputs["name"]

	data = {}
	for ind, val in enumerate(markers):
		data[val] = {
			"chr": chromosome,
			"start": positions[ind], 
			"end": ends[ind], 
			"ppa": None, 
			"gene": gene
		}	

	return data

def write_output(output_path, result):
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	output_return = os.path.join(output_path, "output.pickle")
	with open(output_return, "wb") as output_file:
		pickle.dump(result, output_file)

def write_in_data(output_path, in_data):
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	in_data_return = os.path.join(output_path, "in_data.pickle")
	with open(in_data_return, "wb") as in_data_file:
		pickle.dump(in_data, in_data_file)

def main(io_path, params_path, selection_path, filter_path, overdispersion_path):
	# input_path = os.path.join(target_dir, "input.pickle")
	if selection_path == "all":
		selection = False
	else:
		with open(selection_path, "rb") as selection_file:
			selection = pickle.load(selection_file)

		ind_overdispersion = False
		if overdispersion_path:
			with open(overdispersion_path, "rb") as overdispersion_file:
				overdispersion_dict = pickle.load(overdispersion_file)
			ind_overdispersion = True

		if filter_path == "all":
			snp_filter = False
		else:
			with open(filter_path, "rb") as filter_file:
				snp_filter = pickle.load(filter_file)

		with open(params_path, "rb") as params_file:
			# print(input_path) ####
			params = pickle.load(params_file)

		with open(io_path, "rb") as io_file:
			# print(input_path) ####
			io_data = pickle.load(io_file)

		for input_path, output_path in io_data:
			result = {}
			try:
				with open(input_path, "rb") as input_file:
					# print(input_path) ####
					inputs = pickle.load(input_file, encoding='latin1')
				inputs.update(params)

				inputs["sample_names"] = np.array(inputs["sample_names"])

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
					inputs["sample_names"] = inputs["sample_names"][select]

				num_ppl_raw = np.size(inputs["counts1"])
				# print(num_ppl_raw) ####

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
					inputs["sample_names"] = inputs["sample_names"][threshold]
					# print(np.size(inputs["counts1"])) ####

				# print(inputs["counts1"]) ####
				select_counts = np.logical_and(inputs["counts1"] >= 1, inputs["counts2"] >= 1) 
				
				inputs["hap1"] = inputs["hap1"][select_counts]
				inputs["hap2"] = inputs["hap2"][select_counts]
				inputs["counts1"] = inputs["counts1"][select_counts]
				inputs["counts2"] = inputs["counts2"][select_counts]
				inputs["counts_total"] = inputs["counts_total"][select_counts]
				inputs["sample_names"] = inputs["sample_names"][select_counts]

				if ind_overdispersion:
					default = np.mean(list(overdispersion_dict.values()))
					inputs["overdispersion"] = np.array([overdispersion_dict.get(i, default) for i in inputs["sample_names"]])

				if snp_filter:
					snps_in_filter = [ind for ind, val in enumerate(inputs["snp_ids"]) if val in snp_filter]
					inputs["snp_ids"] = inputs["snp_ids"][snps_in_filter]
					inputs["snp_pos"] = inputs["snp_pos"][snps_in_filter]
					inputs["hap1"] = inputs["hap1"][:, snps_in_filter]
					inputs["hap2"] = inputs["hap2"][:, snps_in_filter]

				# inputs["num_snps_imbalance"] = len(inputs["hap1"])
				# inputs["num_snps_total_exp"] = inputs["num_snps_imbalance"]

				haps_comb = inputs["hap1"] + inputs["hap2"]

				if np.size(inputs["counts1"]) <= 1:
					result["data_error"] = "Insufficient Read Counts"
					write_output(output_path, result)
					return

				# print(haps_comb) ####
				# print(np.logical_not(np.all(haps_comb == haps_comb[0,:], axis=0))) ####
				# print(np.where(np.logical_not(np.all(haps_comb == haps_comb[0,:], axis=0)))) ####
				informative_snps = np.nonzero(np.logical_not(np.all(haps_comb == haps_comb[0,:], axis=0)))[0]
				result["informative_snps"] = informative_snps
				# print(informative_snps) ####

				# inputs["snp_ids"] = inputs["snp_ids"][informative_snps]
				# inputs["snp_pos"] = inputs["snp_pos"][informative_snps]
				inputs["hap1"] = inputs["hap1"][:, informative_snps]
				inputs["hap2"] = inputs["hap2"][:, informative_snps]

				inputs["num_causal_prior"] = inputs["num_causal"]

				if inputs["hap1"].size == 0:
					result["data_error"] = "Insufficient Markers"
					write_output(output_path, result)
					return

				inputs["hap_A"] = inputs["hap1"].astype(np.int)
				inputs["hap_B"] = inputs["hap2"].astype(np.int)

				inputs["counts_A"] = inputs["counts1"].astype(np.int)
				inputs["counts_B"] = inputs["counts2"].astype(np.int)
				inputs["total_exp"] = inputs["counts_total"].astype(np.int)

				if inputs["model_flavors"] == "all":
					model_flavors = set(["full", "indep", "eqtl", "ase", "acav", "fmb"])
				else:
					model_flavors = inputs["model_flavors"]

				result["bed_ctrl"] = get_bed_ctrl(inputs)

				if "full" in model_flavors:
					updates_full = {"num_ppl": None}
					result["causal_set_full"], result["ppas_full"], result["size_probs_full"], model_full = run_model(
						Finemap, inputs, updates_full, informative_snps
					)
					result["ldsr_data_full"] = get_ldsr_data(inputs, result["causal_set_full"], result["ppas_full"])

				if "indep" in model_flavors:
					updates_indep = {"cross_corr_prior": 0., "num_ppl": None}
					result["causal_set_indep"], result["ppas_indep"], result["size_probs_indep"], model_indep = run_model(
						Finemap, inputs, updates_indep, informative_snps
					)
					result["ldsr_data_indep"] = get_ldsr_data(inputs, result["causal_set_indep"], result["ppas_indep"])
					result["z_phi"] = model_indep.imbalance_stats
					result["z_beta"] = model_indep.total_exp_stats

				if "eqtl" in model_flavors:
					updates_eqtl = {"qtl_only": True, "num_ppl": None}
					result["causal_set_eqtl"], result["ppas_eqtl"], result["size_probs_eqtl"], model_eqtl = run_model(
						Finemap, inputs, updates_eqtl, informative_snps
					)
					result["ldsr_data_eqtl"] = get_ldsr_data(inputs, result["causal_set_eqtl"], result["ppas_eqtl"])

				if "ase" in model_flavors:
					updates_ase = {"as_only": True, "num_ppl": None}
					result["causal_set_ase"], result["ppas_ase"], result["size_probs_ase"], model_ase = run_model(
						Finemap, inputs, updates_ase, informative_snps
					)
					result["ldsr_data_ase"] = get_ldsr_data(inputs, result["causal_set_ase"], result["ppas_ase"])

				if "acav" in model_flavors:
					updates_acav = {"num_ppl": None}
					result["causal_set_acav"], result["ppas_acav"], result["size_probs_acav"], model_acav = run_model(
						CaviarASE, inputs, updates_acav, informative_snps
					)
					result["ldsr_data_acav"] = get_ldsr_data(inputs, result["causal_set_acav"], result["ppas_acav"])

				if "cav" in model_flavors:
					updates_cav = {"qtl_only": True, "num_ppl": None}
					result["causal_set_cav"], result["ppas_cav"], result["size_probs_cav"], model_cav = run_model(
						Caviar, inputs, updates_cav, informative_snps
					)
					result["ldsr_data_cav"] = get_ldsr_data(inputs, result["causal_set_cav"], result["ppas_cav"])

				if "fmb" in model_flavors:
					updates_fmb = {"qtl_only": True, "num_ppl": None}
					result["causal_set_fmb"], result["ppas_fmb"], result["size_probs_fmb"], model_rasq = run_model(
						FmBenner, inputs, updates_fmb, informative_snps
					)
					result["ldsr_data_fmb"] = get_ldsr_data(inputs, result["causal_set_fmb"], result["ppas_fmb"])

				write_output(output_path, result)

			except Exception as e:
				trace = traceback.format_exc()
				print(trace, file=sys.stderr)
				message = repr(e)
				result["run_error"] = message
				result["traceback"] = trace
				write_in_data(output_path, inputs)
				write_output(output_path, result)
				return

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

	io_path = sys.argv[1]
	params_path = sys.argv[2]
	selection_path = sys.argv[3]
	filter_path = sys.argv[4]
	overdispersion_path = sys.argv[5]
	main(io_path, params_path, selection_path, filter_path, overdispersion_path)

	
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