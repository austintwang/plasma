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
	import cpickle as pickle
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

print("imp5") ####

def run_model(inputs, input_updates):
	model_inputs = inputs.copy()
	model_inputs.update(input_updates)
	# print(model_inputs) ####

	model = Finemap(**model_inputs)
	model.initialize()

	if inputs["search_mode"] == "exhaustive":
		model.search_exhaustive(inputs["min_causal"], inputs["max_causal"])
	elif inputs["search_mode"] == "shotgun":
		model.search_shotgun(inputs["search_iterations"])

	causal_set = model.get_causal_set(inputs["confidence"])
	ppas = model.get_ppas()

	return causal_set, ppas, model


def main(output_path, input_path, params_path):
	# input_path = os.path.join(target_dir, "input.pickle")
	with open(input_path, "rb") as input_file:
		# print(input_path) ####
		inputs = pickle.load(input_file)

	with open(params_path, "rb") as params_file:
		# print(input_path) ####
		params = pickle.load(params_file)

	inputs.update(params)

	select = np.logical_and(inputs["counts1"] >= 1, inputs["counts2"] >= 1) 

	# num_ppl_raw = np.size(inputs["counts1"])
	# max_ppl = hyperparams.get("max_ppl")
	# if max_ppl and max_ppl < num_ppl_raw:
	# 	threshold = np.array([1] * max_ppl + [0] * (num_ppl_raw - max_ppl))
	# 	np.random.shuffle(threshold)
	# 	select = np.logical_and(select, threshold)
	# 	inputs["num_ppl"] = max_ppl

	inputs["num_snps_imbalance"] = len(inputs["hap1"])
	inputs["num_snps_total_exp"] = inputs["num_snps_imbalance"]

	inputs["hap1"] = np.stack(inputs["hap1"], axis=1)[select]
	inputs["hap2"] = np.stack(inputs["hap2"], axis=1)[select]
	inputs["counts1"] = inputs["counts1"][select]
	inputs["counts2"] = inputs["counts2"][select]
	inputs["counts_total"] = inputs["counts_total"][select]

	num_ppl_raw = np.size(inputs["counts1"])

	if num_ppl_raw == 0:
		print("Insufficient Read Counts")
		return

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

	inputs["num_ppl"] = np.size(inputs["counts1"])
	# print(inputs["num_ppl"]) ####
	# print(max_ppl) ####

	result = {}
	num_ppl = inputs["num_ppl"]
	num_causal = inputs["num_causal"]
	eqtl_herit = 1 - inputs["prop_noise_eqtl"]
	ase_herit = 1 - inputs["prop_noise_ase"]

	coverage = inputs["coverage"]
	overdispersion = inputs["overdispersion"]
	std_fraction = inputs["std_fraction"]
	ase_inherent_var = (np.log(std_fraction) - np.log(1-std_fraction))**2
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

	# print(np.shape(inputs["hap_A"])) ####
	# print(np.shape(inputs["hap_B"])) ####
	# print(np.shape(inputs["counts_A"])) ####
	# print(np.shape(inputs["counts_B"])) ####
	# print(num_ppl) ####
	# raise Exception ####

	updates_full = {
		"corr_stats": corr_stats,
		"imbalance_var_prior": imbalance_var_prior,
		"total_exp_var_prior": total_exp_var_prior
	}
	result["causal_set_full"], result["ppas_full"], model_full = run_model(inputs, updates_full)

	updates_indep = {
		"cross_corr_prior": 0.0, 
		"corr_stats": 0.0,
		"imbalance_var_prior": imbalance_var_prior,
		"total_exp_var_prior": total_exp_var_prior
	}
	result["causal_set_indep"], result["ppas_indep"], model_indep = run_model(inputs, updates_indep)

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
	result["causal_set_eqtl"], result["ppas_eqtl"], model_eqtl = run_model(inputs, updates_eqtl)

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
	result["causal_set_ase"], result["ppas_ase"], model_ase = run_model(inputs, updates_ase)

	model_caviar_ase = EvalCaviarASE(
		model_full, 
		inputs["confidence"], 
		inputs["max_causal"]
	)
	model_caviar_ase.run()
	result["causal_set_caviar_ase"] = model_caviar_ase.causal_set
	result["ppas_caviar_ase"] = model_caviar_ase.post_probs

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	output_return = os.path.join(output_path, "output.pickle")
	with open(output_return, "wb") as output_file:
		pickle.dump(result, output_file)

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
	exit_code = 0
	try:
		# print(os.environ) ####
		# data_dir = os.environ["DATA_DIR"]
		output_path = sys.argv[1]
		input_path = sys.argv[2]
		params_path = sys.argv[3]
		main(output_path, input_path, params_path)
	except Exception as e:
		# print("woiehofwie") ####
		exit_code = 1
		print(str(e), file=sys.stderr)
		print(traceback.format_exc(), file=sys.stderr)
		# raise e
	finally:
		sys.exit(exit_code)
		# pass