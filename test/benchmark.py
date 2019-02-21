from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
from datetime import datetime
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from . import Finemap, SimAse, Haplotypes
from . import EvalCaviar, EvalCaviarASE

def evaluate_bm(targs):
	bm = targs[0]
	simulation = targs[1]
	itr = targs[2]

	result = {}

	model_flavors = bm.params["model_flavors"]
	
	num_ppl = bm.params["num_ppl"]
	if bm.params.get("num_causal", False):
		num_causal = bm.params["max_causal"]
	else:
		num_causal = bm.params["num_causal"]
	eqtl_herit = bm.params["herit_eqtl"]
	ase_herit = bm.params["herit_ase"]

	coverage = bm.params["coverage"]
	overdispersion = bm.params["overdispersion"]
	std_fraction = bm.params["std_fraction"]
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
	# print(correction) ####
	# ase_count_var_simple = ( ####
	# 	2 / coverage
	# 	* (
	# 		1 
	# 		+ (
	# 			1
	# 			/ (
	# 				1 / (np.exp(ase_inherent_var / 2))
	# 			)
	# 		)
	# 	)
	# 	* (1 + overdispersion * (coverage - 1))
	# )
	# print(ase_inherent_var / (ase_inherent_var + ase_count_var_simple)) ####
	# raise(Exception) ####
	ase_herit_adj = ase_herit * correction
	# ase_herit_adj = ase_herit ####

	corr_stats = np.sqrt(
		(num_ppl / num_causal)**2 * eqtl_herit * ase_herit_adj
		/ (
			(1 + eqtl_herit * (num_ppl / num_causal - 1))
			* (1 + ase_herit_adj * (num_ppl / num_causal - 1))
		)
	)
	# print(corr_stats) ####
	iv = (
		(num_ppl / num_causal * ase_herit_adj / (1 - ase_herit_adj)) 
	)
	xv = (
		(num_ppl / num_causal * eqtl_herit / (1 - eqtl_herit)) 
	)
	# unbias = lambda x: x * np.log(
	# 	1
	# 	+ x * (2 * x + 1) / (2 * (x + 1))
	# 	+ x**2 * (3 * x + 1) / (3 * (x + 1)**2)
	# 	+ x**3 * (2 * (2 * x + 1)**2 + 48 * (4 * x + 1)) / (192 * (x + 1)**3)
	# )
	# unbias = lambda x: x * 40
	# imbalance_var_prior = unbias(iv)
	# total_exp_var_prior = unbias(xv)

	imbalance_var_prior = 1 * iv
	total_exp_var_prior = 1 * xv

	print("\nIteration {0}".format(str(itr + 1)))
	# print("Generating Simulation Data")
	# simulation = bm.simulation.generate_data()
	# sim_result = {
	# 	"counts_A": bm.simulation.counts_A,
	# 	"counts_B": bm.simulation.counts_B,
	# 	"total_exp": bm.simulation.total_exp,
	# 	"hap_A": bm.simulation.hap_A,
	# 	"hap_B": bm.simulation.hap_B
	# }
	sim_result = {
		"counts_A": simulation["counts_A"],
		"counts_B": simulation["counts_B"],
		"total_exp": simulation["total_exp"],
		"hap_A": simulation["hap_A"],
		"hap_B": simulation["hap_B"]
	}
	causal_config = simulation["causal_config"]
	# print(causal_config) ####
	# print("Finished Generating Simulation Data")

	# print(sim_result["hap_A"].tolist()) ####
	# print(sim_result["hap_B"].tolist()) ####
	# null = tuple([0] * bm.params["num_snps"]) ####
	# print(sim_result["counts_A"]) ####

	model_inputs = bm.model_params.copy()
	model_inputs.update(sim_result)
	if "full" in model_flavors:
		# print("Initializing Full Model")
		model_inputs_full = model_inputs.copy()
		model_inputs_full.update({
			"corr_stats": corr_stats,
			"imbalance_var_prior": imbalance_var_prior,
			"total_exp_var_prior": total_exp_var_prior
		})
		# print(model_inputs) ####
		# print(model_inputs_full["counts_A"]) ####
		model_full = Finemap(**model_inputs_full)
		model_full.initialize()
		# print("Finished Initializing Full Model")
		# print("Starting Search")
		if bm.params["search_mode"] == "exhaustive":
			model_full.search_exhaustive(bm.params["min_causal"], bm.params["max_causal"])
		elif bm.params["search_mode"] == "shotgun":
			model_full.search_shotgun(bm.params["search_iterations"])
		# print("Finished Search Under Full Model")

		causal_set = model_full.get_causal_set(bm.params["confidence"])
		assert all([i == 0 or i == 1 for i in causal_set])
		causal_set_size = sum(causal_set)
		result["set_sizes_full"] = causal_set_size
		# print(causal_set_size) ####
		# print(model_full.get_probs()[tuple(causal_config)]) ####
		# print(model_full.get_probs()[null]) ####
		x = model_full.imbalance_stats
		result["max_stat_ase_full"] = abs(max(x.min(), x.max(), key=abs) )

		recall = bm._recall(causal_set, causal_config)
		# for ind, val in enumerate(causal_config):
		# 	if val == 1:
		# 		if causal_set[ind] != 1:
		# 			recall = 0
		result["recall_full"] = recall
		# print(recall) ####
		# print(model_full.get_probs_sorted()[:10]) ####

		result["inclusions_full"] = bm._inclusion(model_full.get_ppas(), causal_config)
		# print(model_full.get_probs()) ####

	if "indep" in model_flavors:
		# print("Initializing Independent Model")
		model_inputs_indep = model_inputs.copy()
		model_inputs_indep.update({
			"cross_corr_prior": 0.0, 
			"corr_stats": 0.0,
			"imbalance_var_prior": imbalance_var_prior,
			"total_exp_var_prior": total_exp_var_prior
		})
		# print("Finished Initializing Independent Model")
		# print("Starting Search Under Independent Model")
		# print(model_inputs_indep["counts_A"]) ####
		model_indep = Finemap(**model_inputs_indep)
		model_indep.initialize()
		if bm.params["search_mode"] == "exhaustive":
			model_indep.search_exhaustive(bm.params["min_causal"], bm.params["max_causal"])
		elif bm.params["search_mode"] == "shotgun":
			model_indep.search_shotgun(bm.params["search_iterations"])
		# print("Finished Search Under Independent Model")

		causal_set_indep = model_indep.get_causal_set(bm.params["confidence"])
		# assert all([i == 0 or i == 1 for i in causal_set_indep])
		causal_set_indep_size = sum(causal_set_indep)
		result["set_sizes_indep"] = causal_set_indep_size
		# print(causal_set_indep_size) ####
		# print(model_eqtl.get_probs_sorted()) ####
		# model_eqtl.get_probs_sorted() ####
		# print(model_indep.get_probs()[tuple(causal_config)]) ####
		# print(model_indep.get_probs()[null]) ####
		x = model_indep.imbalance_stats
		result["max_stat_ase_indep"] = abs(max(x.min(), x.max(), key=abs)) 

		recall = bm._recall(causal_set_indep, causal_config)
		# for ind, val in enumerate(causal_config):
		# 	if val == 1:
		# 		if causal_set_indep[ind] != 1:
		# 			recall = 0
		result["recall_indep"] = recall
		# print(recall) ####

		result["inclusions_indep"] = bm._inclusion(model_indep.get_ppas(), causal_config)


	if "eqtl" in model_flavors:
		# print("Initializing eQTL Model")
		model_inputs_eqtl = model_inputs.copy()
		model_inputs_eqtl.update({
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
		})
		# print("Finished Initializing eQTL Model")
		# print("Starting Search Under eQTL Model")
		model_eqtl = Finemap(**model_inputs_eqtl)
		model_eqtl.initialize()
		if bm.params["search_mode"] == "exhaustive":
			model_eqtl.search_exhaustive(bm.params["min_causal"], bm.params["max_causal"])
		elif bm.params["search_mode"] == "shotgun":
			model_eqtl.search_shotgun(bm.params["search_iterations"])
		# print("Finished Search Under eQTL Model")

		causal_set_eqtl = model_eqtl.get_causal_set(bm.params["confidence"])
		assert all([i == 0 or i == 1 for i in causal_set_eqtl])
		causal_set_eqtl_size = sum(causal_set_eqtl)
		result["set_sizes_eqtl"] = causal_set_eqtl_size
		# print(causal_set_eqtl_size) ####
		# print(model_eqtl.get_probs_sorted()) ####
		# print(model_eqtl.get_probs_sorted()) ####
		# print(model_eqtl.get_probs()[tuple(causal_config)]) ####
		# print(model_eqtl.get_probs()[tuple(null)]) ####
		# ppas = model_eqtl.get_ppas() ####
		# np.savetxt("ppas_eqtl.txt", np.array(ppas)) ####
		# print(model_eqtl.total_exp_stats[causal_config == True][0]) ####
		# eqtl_cstats.append(model_eqtl.total_exp_stats[causal_config == True][0]) ####

		recall = bm._recall(causal_set_eqtl, causal_config)
		# for ind, val in enumerate(causal_config):
		# 	if val == 1:
		# 		if causal_set_eqtl[ind] != 1:
		# 			recall = 0
		result["recall_eqtl"] = recall
		# print(recall) ####
		# ppa_eqtl = model_eqtl.get_ppas() ####

		result["inclusions_eqtl"] = bm._inclusion(model_eqtl.get_ppas(), causal_config)

	if "ase" in model_flavors:
		# print("Initializing ASE Model")
		model_inputs_ase = model_inputs.copy()
		model_inputs_ase.update({
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
		})
		# print("Finished Initializing ASE Model")
		# print("Starting Search Under ASE Model")
		model_ase = Finemap(**model_inputs_ase)
		model_ase.initialize()
		if bm.params["search_mode"] == "exhaustive":
			model_ase.search_exhaustive(bm.params["min_causal"], bm.params["max_causal"])
		elif bm.params["search_mode"] == "shotgun":
			model_ase.search_shotgun(bm.params["search_iterations"])
		# print("Finished Search Under ASE Model")

		causal_set_ase = model_ase.get_causal_set(bm.params["confidence"])
		assert all([i == 0 or i == 1 for i in causal_set_ase])
		causal_set_ase_size = sum(causal_set_ase)
		result["set_sizes_ase"] = causal_set_ase_size
		# print(causal_set_ase_size) ####
		# print(model_eqtl.get_probs_sorted()) ####
		# model_eqtl.get_probs_sorted() ####
		# print(model_ase.get_probs()[tuple(causal_config)]) ####
		# print(model_ase.get_probs()[tuple(null)]) ####
		x = model_ase.imbalance_stats
		result["max_stat_ase_ase"] = abs(max(x.min(), x.max(), key=abs)) 

		recall = bm._recall(causal_set_ase, causal_config)
		# for ind, val in enumerate(causal_config):
		# 	if val == 1:
		# 		if causal_set_ase[ind] != 1:
		# 			recall = 0
		result["recall_ase"] = recall
		# print(recall) ####

		result["inclusions_ase"] = bm._inclusion(model_ase.get_ppas(), causal_config)

	# if causal_set_ase_size == 181: ####
	# 	ps = model_ase.get_probs_sorted()
	# 	with open("null_ase.txt", "w") as null_ase:
	# 		null_ase.write("\n".join("\t".join(str(j) for j in i) for i in ps))
	# 	# np.savetxt("null_ase.txt", np.array(model_ase.get_probs_sorted()))
	# 	raise Exception

	if "cav" in model_flavors:
		model_inputs_dummy = model_inputs.copy()
		model_inputs_dummy.update({
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
		})
		# print("Finished Initializing eQTL Model")
		# print("Starting Search Under eQTL Model")
		model_dummy = Finemap(**model_inputs_dummy)
		model_dummy.initialize()
		model_caviar = EvalCaviar(
			model_dummy, 
			bm.params["confidence"], 
			bm.params["max_causal"]
		)
		model_caviar.run()
		causal_set_caviar = model_caviar.causal_set
		causal_set_caviar_size = sum(causal_set_caviar)
		result["set_sizes_caviar"] = causal_set_caviar_size
		recall = bm._recall(causal_set_caviar, causal_config)
		result["recall_caviar"] = recall
		# ppa_caviar = model_caviar.post_probs ####

		result["inclusions_caviar"] = bm._inclusion(model_caviar.post_probs, causal_config)

	if "acav" in model_flavors:
		model_inputs_dummy = model_inputs.copy()
		model_inputs_indep.update({
			"cross_corr_prior": 0.0, 
			"corr_stats": 0.0,
			"imbalance_var_prior": imbalance_var_prior,
			"total_exp_var_prior": total_exp_var_prior
		})
		# print("Finished Initializing eQTL Model")
		# print("Starting Search Under eQTL Model")
		model_dummy = Finemap(**model_inputs_dummy)
		model_dummy.initialize()
		model_caviar_ase = EvalCaviarASE(
			model_dummy, 
			bm.params["confidence"], 
			bm.params["max_causal"]
		)
		model_caviar_ase.run()
		causal_set_caviar_ase = model_caviar_ase.causal_set
		causal_set_caviar_size_ase = sum(causal_set_caviar_ase)
		result["set_sizes_caviar_ase"] = causal_set_caviar_size_ase
		recall = bm._recall(causal_set_caviar_ase, causal_config)
		result["recall_caviar_ase"] = recall

		result["inclusions_caviar_ase"] = bm._inclusion(model_caviar_ase.post_probs, causal_config)

	# print(result["inclusions_caviar_ase"]) ####

	# print("\n".join(str(i) for i in zip(ppa_eqtl, ppa_caviar))) ####
	# raise Exception ####

	return result


class Benchmark(object):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	res_path = os.path.join("results")
	num_cpu = multiprocessing.cpu_count()
	def __init__(self, params):
		self.params = params
		self.haplotypes = Haplotypes(params)

		self.results = []
		self.results_df = None
		self.primary_var_vals = []
		self.simulation = SimAse(self)

		self.update_model_params()
		self.update_sim_params()

		# self.time = datetime.now()
		# self.timestamp = self.time.strftime("%y%m%d%H%M%f")
		self.counter = 0
		self.test_count = self.params["test_count"]
		self.count_digits = len(str(self.test_count))

		# self.output_folder = self.timestamp + "_" + self.params["test_name"]
		# self.output_path = os.path.join(self.dir_path, self.res_path, self.output_folder)

		self.output_folder = self.params["test_name"]
		self.test_path = self.params["test_path"]
		self.output_path = os.path.join(self.test_path, self.output_folder)

	def update_model_params(self):
		self.model_params = {
			"num_snps_imbalance": self.params["num_snps"],
			"num_snps_total_exp": self.params["num_snps"],
			"num_ppl_imbalance": self.params["num_ppl"],
			"num_ppl_total_exp": self.params["num_ppl"],
			"overdispersion": self.params["overdispersion"]
		}

	def update_sim_params(self):
		self.sim_params = {
			"num_snps": self.params["num_snps"],
			"num_ppl": self.params["num_ppl"],
			# "var_effect_size": self.params["var_effect_size"],
			"overdispersion": self.params["overdispersion"],
			"herit_eqtl": self.params["herit_eqtl"],
			"herit_ase": self.params["herit_ase"],
			# "baseline_exp": self.params["baseline_exp"],
			"num_causal": self.params["num_causal"],
			# "ase_read_prop": self.params["ase_read_prop"],
			"std_fraction": self.params["std_fraction"],
			"coverage": self.params["coverage"]
		}
		self.simulation.update()


	def output_result(self, result, out_dir):
		if self.params["model_flavors"] == "all":
			model_flavors = set(["full", "indep", "eqtl", "ase", "cav", "acav"])
		else:
			model_flavors = self.params["model_flavors"]

		title_var = self.params["primary_var_display"]
		var_value = str(self.params[self.params["primary_var"]])
		num_snps = self.params["num_snps"]

		recall_list = []

		if "full" in model_flavors:
			set_sizes_full = result["set_sizes_full"]
			recall_rate_full = result["recall_rate_full"]
			inclusion_rate_full = list(result["inclusion_rate_full"])
			recall_list.append("Joint-Correlated:{:>15}\n".format(recall_rate_full))
			with open(os.path.join(out_dir, "causal_set_sizes.txt"), "w") as cssfull:
				cssfull.write("\n".join(str(i) for i in set_sizes_full))

		if "indep" in model_flavors:
			set_sizes_indep = result["set_sizes_indep"]
			recall_rate_indep = result["recall_rate_indep"]
			inclusion_rate_indep = list(result["inclusion_rate_indep"])
			recall_list.append("Joint-Independent:{:>15}\n".format(recall_rate_indep))
			with open(os.path.join(out_dir, "causal_set_sizes_independent.txt"), "w") as cssindep:
				cssindep.write("\n".join(str(i) for i in set_sizes_indep))

		if "eqtl" in model_flavors:
			set_sizes_eqtl = result["set_sizes_eqtl"]
			recall_rate_eqtl = result["recall_rate_eqtl"]
			inclusion_rate_eqtl = list(result["inclusion_rate_eqtl"])
			recall_list.append("eQTL-Only:{:>15}\n".format(recall_rate_eqtl))
			with open(os.path.join(out_dir, "causal_set_sizes_eqtl_only.txt"), "w") as csseqtl:
				csseqtl.write("\n".join(str(i) for i in set_sizes_eqtl))

		if "ase" in model_flavors:
			set_sizes_ase = result["set_sizes_ase"]
			recall_rate_ase = result["recall_rate_ase"]
			inclusion_rate_ase = list(result["inclusion_rate_ase"])
			recall_list.append("ASE-Only:{:>15}\n".format(recall_rate_ase))
			with open(os.path.join(out_dir, "causal_set_sizes_ase_only.txt"), "w") as cssase:
				cssase.write("\n".join(str(i) for i in set_sizes_ase))

		if "cav" in model_flavors:
			set_sizes_caviar = result["set_sizes_caviar"]
			recall_rate_caviar = result["recall_rate_caviar"]
			inclusion_rate_caviar = list(result["inclusion_rate_caviar"])
			recall_list.append("CAVIAR:{:>15}\n".format(recall_rate_caviar))
			with open(os.path.join(out_dir, "causal_set_sizes_caviar.txt"), "w") as csscav:
				csscav.write("\n".join(str(i) for i in set_sizes_caviar))

		if "acav" in model_flavors:
			set_sizes_caviar_ase = result["set_sizes_caviar_ase"]
			recall_rate_caviar_ase = result["recall_rate_caviar_ase"]
			inclusion_rate_caviar_ase = list(result["inclusion_rate_caviar_ase"])
			recall_list.append("CAVIAR-ASE:{:>15}\n".format(recall_rate_caviar_ase))
			with open(os.path.join(out_dir, "causal_set_sizes_caviar_ase.txt"), "w") as csscavase:
				csscavase.write("\n".join(str(i) for i in set_sizes_caviar_ase))

		params_str = "\n".join("{:<20}{:>20}".format(k, v) for k, v in self.params.viewitems())
		with open(os.path.join(out_dir, "parameters.txt"), "w") as params_file:
			params_file.write(params_str)

		with open(os.path.join(out_dir, "recalls.txt"), "w") as rr:
			rr.writelines(recall_list)

		sns.set(style="dark", font="Roboto")
		if "full" in model_flavors:
			try:
				sns.distplot(
					set_sizes_full,
					hist=False,
					kde=True,
					kde_kws={"linewidth": 2, "shade":False},
					label="Joint-Correlated"
				)
			except Exception:
				pass
		if "indep" in model_flavors:
			try:
				sns.distplot(
					set_sizes_indep,
					hist=False,
					kde=True,
					kde_kws={"linewidth": 2, "shade":False},
					label="Joint-Independent"			
				)
			except Exception:
				pass
		if "eqtl" in model_flavors:
			try:
				sns.distplot(
					set_sizes_eqtl,
					hist=False,
					kde=True,
					kde_kws={"linewidth": 2, "shade":False},
					label="eQTL-Only"			
				)
			except Exception:
				pass
		if "ase" in model_flavors:
			try:
				sns.distplot(
					set_sizes_ase,
					hist=False,
					kde=True,
					kde_kws={"linewidth": 2, "shade":False},
					label="ASE-Only"			
				)
			except Exception:
				pass
		if "cav" in model_flavors:
			try:
				sns.distplot(
					set_sizes_caviar,
					hist=False,
					kde=True,
					kde_kws={"linewidth": 2, "shade":False},
					label="ASE-Only"			
				)
			except Exception:
				pass
		if "acav" in model_flavors:
			try:
				sns.distplot(
					set_sizes_caviar_ase,
					hist=False,
					kde=True,
					kde_kws={"linewidth": 2, "shade":False},
					label="CAVIAR-ASE"			
				)
			except Exception:
				pass

		plt.xlim(0, None)
		plt.legend(title="Model")
		plt.xlabel("Set Size")
		plt.ylabel("Density")
		plt.title("Distribution of Causal Set Sizes, {0} = {1}".format(title_var, var_value))
		plt.savefig(os.path.join(out_dir, "set_size_distribution.svg"))
		plt.clf()

		# try:
		# 	sns.set(style="whitegrid", font="Roboto")
		# 	sns.distplot(
		# 		set_sizes_full,
		# 		hist=False,
		# 		kde=True,
		# 		kde_kws={"linewidth": 2, "shade":False},
		# 		label="Joint-Correlated"
		# 	)
		# 	sns.distplot(
		# 		set_sizes_indep,
		# 		hist=False,
		# 		kde=True,
		# 		kde_kws={"linewidth": 2, "shade":False},
		# 		label="Joint-Independent"
		# 	)
		# 	sns.distplot(
		# 		set_sizes_eqtl,
		# 		hist=False,
		# 		kde=True,
		# 		kde_kws={"linewidth": 3, "shade":True},
		# 		label="eQTL-Only"
		# 	)
		# 	sns.distplot(
		# 		set_sizes_ase,
		# 		hist=False,
		# 		kde=True,
		# 		kde_kws={"linewidth": 3, "shade":True},
		# 		label="ASE-Only"
		# 	)
		# 	sns.distplot(
		# 		set_sizes_caviar,
		# 		hist=False,
		# 		kde=True,
		# 		kde_kws={"linewidth": 3, "shade":True},
		# 		label="CAVIAR"
		# 	)
		# 	sns.distplot(
		# 		set_sizes_caviar_ase,
		# 		hist=False,
		# 		kde=True,
		# 		kde_kws={"linewidth": 3, "shade":True},
		# 		label="CAVIAR-ASE"
		# 	)
		# 	plt.xlim(0, None)
		# 	plt.legend(title="Model")
		# 	plt.xlabel("Set Size")
		# 	plt.ylabel("Density")
		# 	plt.title("Distribution of Causal Set Sizes, {0} = {1}".format(title_var, var_value))
		# 	plt.savefig(os.path.join(out_dir, "set_size_distribution.svg"))
		# 	plt.clf()
		# except Exception:
		# 	# raise ####
		# 	plt.clf()

		inclusions_dict = {
			"Number of Selected Markers": [],
			"Inclusion Rate": [],
			"Model": []
		}

		if "full" in model_flavors:
			inclusions_dict["Number of Selected Markers"].extend(range(1, num_snps+1))
			inclusions_dict["Inclusion Rate"].extend(inclusion_rate_full)
			inclusions_dict["Model"].extend(num_snps * ["Joint-Correlated"])

		if "indep" in model_flavors:
			inclusions_dict["Number of Selected Markers"].extend(range(1, num_snps+1))
			inclusions_dict["Inclusion Rate"].extend(inclusion_rate_indep)
			inclusions_dict["Model"].extend(num_snps * ["Joint-Independent"])

		if "eqtl" in model_flavors:
			inclusions_dict["Number of Selected Markers"].extend(range(1, num_snps+1))
			inclusions_dict["Inclusion Rate"].extend(inclusion_rate_eqtl)
			inclusions_dict["Model"].extend(num_snps * ["eQTL-Only"])

		if "ase" in model_flavors:
			inclusions_dict["Number of Selected Markers"].extend(range(1, num_snps+1))
			inclusions_dict["Inclusion Rate"].extend(inclusion_rate_ase)
			inclusions_dict["Model"].extend(num_snps * ["ASE-Only"])

		if "cav" in model_flavors:
			inclusions_dict["Number of Selected Markers"].extend(range(1, num_snps+1))
			inclusions_dict["Inclusion Rate"].extend(inclusion_rate_caviar)
			inclusions_dict["Model"].extend(num_snps * ["CAVIAR"])

		if "acav" in model_flavors:
			inclusions_dict["Number of Selected Markers"].extend(range(1, num_snps+1))
			inclusions_dict["Inclusion Rate"].extend(inclusion_rate_caviar_ase)
			inclusions_dict["Model"].extend(num_snps * ["CAVIAR-ASE"])

		# print(inclusions_dict) ####
		# print(len(inclusion_rate_full)) ####
		# print(num_snps) ####
		inclusions_df = pd.DataFrame(inclusions_dict)

		sns.set(font="Roboto")
		sns.lineplot(x="Number of Selected Markers", y="Inclusion Rate", hue="Model", data=inclusions_df)
		plt.title("Inclusion Rate vs. Selection Size, {0} = {1}".format(title_var, var_value))
		plt.savefig(os.path.join(out_dir, "inclusion.svg"))
		plt.clf()


	def output_summary(self):
		if self.params["model_flavors"] == "all":
			model_flavors = set(["full", "indep", "eqtl", "ase", "cav", "acav"])
		else:
			model_flavors = self.params["model_flavors"]
		title_var = self.params["primary_var_display"]
		num_trials = self.test_count

		rec_dict = {
			title_var: [],
			"Recall Rate": [],
			"Model": []
		}

		if "full" in model_flavors:
			recall_rate_full = [i["recall_rate_full"] for i in self.results]
			rec_dict[title_var].extend(self.primary_var_vals)
			rec_dict["Recall Rate"].extend(recall_rate_full)
			rec_dict["Model"].extend(num_trials * ["Joint-Correlated"])

		if "indep" in model_flavors:
			recall_rate_indep = [i["recall_rate_indep"] for i in self.results]
			rec_dict[title_var].extend(self.primary_var_vals)
			rec_dict["Recall Rate"].extend(recall_rate_indep)
			rec_dict["Model"].extend(num_trials * ["Joint-Independent"])

		if "eqtl" in model_flavors:
			recall_rate_eqtl = [i["recall_rate_eqtl"] for i in self.results]
			rec_dict[title_var].extend(self.primary_var_vals)
			rec_dict["Recall Rate"].extend(recall_rate_eqtl)
			rec_dict["Model"].extend(num_trials * ["eQTL-Only"])

		if "ase" in model_flavors:
			recall_rate_ase = [i["recall_rate_ase"] for i in self.results]
			rec_dict[title_var].extend(self.primary_var_vals)
			rec_dict["Recall Rate"].extend(recall_rate_ase)
			rec_dict["Model"].extend(num_trials * ["ASE-Only"])

		if "cav" in model_flavors:
			recall_rate_caviar = [i["recall_rate_caviar"] for i in self.results]
			rec_dict[title_var].extend(self.primary_var_vals)
			rec_dict["Recall Rate"].extend(recall_rate_caviar)
			rec_dict["Model"].extend(num_trials * ["CAVIAR"])

		if "acav" in model_flavors:
			recall_rate_caviar_ase = [i["recall_rate_caviar_ase"] for i in self.results]
			rec_dict[title_var].extend(self.primary_var_vals)
			rec_dict["Recall Rate"].extend(recall_rate_caviar_ase)
			rec_dict["Model"].extend(num_trials * ["CAVIAR-ASE"])

		# print(rec_dict) ####
		rec_df = pd.DataFrame(rec_dict)

		sns.set(font="Roboto")
		sns.lmplot(title_var, "Recall Rate", rec_df, hue="Model")
		# sns.lmplot(self.primary_var_vals, recall_full)
		# sns.lmplot(self.primary_var_vals, recall_indep)
		# sns.lmplot(self.primary_var_vals, recall_eqtl)
		# sns.lmplot(self.primary_var_vals, recall_ase)
		# plt.legend(title="Model")
		# plt.xlabel(title_var)
		# plt.ylabel("Recall Rate")
		plt.title("Recall Rates Across {0}".format(title_var))
		plt.savefig(os.path.join(self.output_path, "recalls.svg"))
		plt.clf()

		# dflst = []
		# for ind, dct in enumerate(self.results):
		# 	var_value = self.primary_var_vals[ind]
		# 	for i in dct["set_sizes_full"]:
		# 		dflst.append([i, var_value, "Full"])
		# 	for i in dct["set_sizes_eqtl"]:
		# 		dflst.append([i, var_value, "eQTL-Only"])
		# res_df = pd.DataFrame(dflst, columns=["Set Size", title_var, "Model"])

		dflst = []
		for ind, dct in enumerate(self.results):
			var_value = self.primary_var_vals[ind]
			if "full" in model_flavors:
				for i in dct["set_sizes_full"]:
					dflst.append([i, var_value, "Joint-Correlated"])
			if "indep" in model_flavors:
				for i in dct["set_sizes_indep"]:
					dflst.append([i, var_value, "Joint-Independent"])
			if "eqtl" in model_flavors:
				for i in dct["set_sizes_eqtl"]:
					dflst.append([i, var_value, "eQTL-Only"])
			if "ase" in model_flavors:
				for i in dct["set_sizes_ase"]:
					dflst.append([i, var_value, "ASE-Only"])
			if "cav" in model_flavors:
				for i in dct["set_sizes_caviar"]:
					dflst.append([i, var_value, "CAVIAR"])
			if "acav" in model_flavors:
				for i in dct["set_sizes_caviar_ase"]:
					dflst.append([i, var_value, "CAVIAR-ASE"])

		res_df = pd.DataFrame(dflst, columns=["Set Size", title_var, "Model"])
		
		sns.set(style="whitegrid", font="Roboto")
		sns.violinplot(
			x=title_var,
			y="Set Size",
			hue="Model",
			data=res_df,
		)
		
		# sns.set(style="whitegrid", font="Roboto")
		# sns.violinplot(
		# 	x=title_var,
		# 	y="Set Size",
		# 	hue="Model",
		# 	data=res_df,
		# 	split=True,
		# 	inner="quartile"
		# )
		plt.title("Causal Set Sizes across {0}".format(title_var))
		plt.savefig(os.path.join(self.output_path, "causal_sets.svg"))
		plt.clf()

	def set_output_folder(self):
		count_str = str(self.counter + 1).zfill(self.count_digits)
		test_folder = "{0}_{1}_{2}".format(
			count_str, 
			self.params["primary_var"], 
			str(self.params[self.params["primary_var"]])
		)
		test_path = os.path.join(self.output_path, test_folder)
		os.makedirs(test_path)
		self.counter += 1
		return test_path
	
	@staticmethod
	def _recall(causal_set, causal_config):
		recall = 1
		for ind, val in enumerate(causal_config):
			if val == 1:
				if causal_set[ind] != 1:
					recall = 0
		return recall

	@staticmethod
	def _inclusion(ppas, causal_config):
		selections = np.flip(np.argsort(ppas))
		causals = causal_config[selections]
		inclusions = np.cumsum(causals) / np.sum(causal_config)
		return inclusions
	
	def test(self, **kwargs):
		# count_str = str(self.counter + 1).zfill(self.count_digits)
		# test_folder = "{0}_{1}_{2}".format(
		# 	count_str, 
		# 	self.params["primary_var"], 
		# 	str(self.params[self.params["primary_var"]])
		# )
		# test_path = os.path.join(self.output_path, test_folder)
		# os.makedirs(test_path)

		for k, v in kwargs.viewitems():
			self.params[k] = v
		self.update_model_params()
		self.update_sim_params()

		model_flavors = self.params["model_flavors"]

		result = {
			"set_sizes_full": [],
			"set_sizes_indep": [],
			"set_sizes_eqtl": [],
			"set_sizes_ase": [],
			"set_sizes_caviar": [],
			"set_sizes_caviar_ase": [],
			"recall_full": [],
			"recall_indep": [],
			"recall_eqtl": [],
			"recall_ase": [],
			"recall_caviar": [],
			"recall_caviar_ase": [],
		}
		
		# num_ppl = self.params["num_ppl"]
		# eqtl_herit = 1 - self.params["prop_noise_eqtl"]
		# ase_herit = 1 - self.params["prop_noise_ase"]

		# coverage = self.params["coverage"]
		# overdispersion = self.params["overdispersion"]
		# std_fraction = self.params["std_fraction"]
		# ase_inherent_var = (np.log(std_fraction) - np.log(1-std_fraction))**2
		# ase_count_var = (
		# 	2 / coverage
		# 	* (
		# 		1 
		# 		+ (
		# 			1
		# 			/ (
		# 				1 / (np.exp(ase_inherent_var / 2))
		# 				+ 1 / (np.exp(ase_inherent_var / 2)**3)
		# 				* (
		# 					(np.exp(ase_inherent_var * 2) + 1) / 2
		# 					- np.exp(ase_inherent_var)
		# 				)
		# 			)
		# 		)
		# 	)
		# 	* (1 + overdispersion * (coverage - 1))
		# )
		# correction = ase_inherent_var / (ase_inherent_var + ase_count_var)
		# # print(correction) ####
		# # ase_count_var_simple = ( ####
		# # 	2 / coverage
		# # 	* (
		# # 		1 
		# # 		+ (
		# # 			1
		# # 			/ (
		# # 				1 / (np.exp(ase_inherent_var / 2))
		# # 			)
		# # 		)
		# # 	)
		# # 	* (1 + overdispersion * (coverage - 1))
		# # )
		# # print(ase_inherent_var / (ase_inherent_var + ase_count_var_simple)) ####
		# # raise(Exception) ####
		# ase_herit_adj = ase_herit * correction
		# # ase_herit_adj = ase_herit ####

		# corr_stats = np.sqrt(
		# 	num_ppl**2 * eqtl_herit * ase_herit_adj
		# 	/ (
		# 		(1 + eqtl_herit * (num_ppl - 1))
		# 		* (1 + ase_herit_adj * (num_ppl - 1))
		# 	)
		# )
		# # print(corr_stats) ####
		# iv = (
		# 	(num_ppl * ase_herit_adj / (1 - ase_herit_adj)) 
		# )
		# xv = (
		# 	(num_ppl * eqtl_herit / (1 - eqtl_herit)) 
		# )
		# # unbias = lambda x: x * np.log(
		# # 	1
		# # 	+ x * (2 * x + 1) / (2 * (x + 1))
		# # 	+ x**2 * (3 * x + 1) / (3 * (x + 1)**2)
		# # 	+ x**3 * (2 * (2 * x + 1)**2 + 48 * (4 * x + 1)) / (192 * (x + 1)**3)
		# # )
		# unbias = lambda x: x
		# imbalance_var_prior = unbias(iv)
		# total_exp_var_prior = unbias(xv)

		# print(np.sqrt(total_exp_var_prior)) ####
		# print(np.sqrt(imbalance_var_prior)) ####
		# raise ####
		
		# eqtl_cstats = [] ####
		num_workers = self.num_cpu - 1
		# num_workers = 1 ####
		trials = multiprocessing.Pool(num_workers)
		targ_list = [
			(self, self.simulation.generate_data(), i,) 
			for i in xrange(self.params["iterations"])
		]
		trial_results = trials.map(evaluate_bm, targ_list)
		trials.close()
		
		# print(trial_results) ####
		for i in trial_results:
			for k, v in i.viewitems():
				result.setdefault(k, []).append(v)

		# for itr in xrange(self.params["iterations"]):
		# 	print("\nIteration {0}".format(str(itr + 1)))
		# 	# print("Generating Simulation Data")
		# 	self.simulation.generate_data()
		# 	sim_result = {
		# 		"counts_A": self.simulation.counts_A,
		# 		"counts_B": self.simulation.counts_B,
		# 		"total_exp": self.simulation.total_exp,
		# 		"hap_A": self.simulation.hap_A,
		# 		"hap_B": self.simulation.hap_B
		# 	}
		# 	causal_config = self.simulation.causal_config
		# 	# print(causal_config) ####
		# 	# print("Finished Generating Simulation Data")

		# 	# print(sim_result["hap_A"].tolist()) ####
		# 	# print(sim_result["hap_B"].tolist()) ####
		# 	# null = tuple([0] * self.params["num_snps"]) ####

		# 	# print("Initializing Full Model")
		# 	model_inputs = self.model_params.copy()
		# 	model_inputs.update(sim_result)
		# 	model_inputs.update({
		# 		"corr_stats": corr_stats,
		# 		"imbalance_var_prior": imbalance_var_prior,
		# 		"total_exp_var_prior": total_exp_var_prior
		# 	})
		# 	# print(model_inputs) ####
		# 	model_full = Finemap(**model_inputs)
		# 	model_full.initialize()
		# 	# print("Finished Initializing Full Model")
		# 	# print("Starting Search")
		# 	if self.params["search_mode"] == "exhaustive":
		# 		model_full.search_exhaustive(self.params["min_causal"], self.params["max_causal"])
		# 	elif self.params["search_mode"] == "shotgun":
		# 		model_full.search_shotgun(self.params["search_iterations"])
		# 	# print("Finished Search Under Full Model")

		# 	causal_set = model_full.get_causal_set(self.params["confidence"])
		# 	assert all([i == 0 or i == 1 for i in causal_set])
		# 	causal_set_size = sum(causal_set)
		# 	result["set_sizes_full"].append(causal_set_size)
		# 	# print(causal_set_size) ####
		# 	# print(model_full.get_probs()[tuple(causal_config)]) ####
		# 	# print(model_full.get_probs()[null]) ####
		# 	x = model_full.imbalance_stats
		# 	result["max_stat_ase_full"] = abs(max(x.min(), x.max(), key=abs) )


		# 	recall = self._recall(causal_set, causal_config)
		# 	# for ind, val in enumerate(causal_config):
		# 	# 	if val == 1:
		# 	# 		if causal_set[ind] != 1:
		# 	# 			recall = 0
		# 	result["recall_full"].append(recall)
		# 	# print(recall) ####
		# 	# print(model_full.get_probs_sorted()[:10]) ####


		# 	# print("Initializing Independent Model")
		# 	model_inputs_indep = model_inputs.copy()
		# 	model_inputs_indep.update({
		# 		"cross_corr_prior": 0.0, 
		# 		"corr_stats": 0.0,
		# 		"imbalance_var_prior": imbalance_var_prior,
		# 		"total_exp_var_prior": total_exp_var_prior
		# 	})
		# 	# print("Finished Initializing Independent Model")
		# 	# print("Starting Search Under Independent Model")
		# 	model_indep = Finemap(**model_inputs_indep)
		# 	model_indep.initialize()
		# 	if self.params["search_mode"] == "exhaustive":
		# 		model_indep.search_exhaustive(self.params["min_causal"], self.params["max_causal"])
		# 	elif self.params["search_mode"] == "shotgun":
		# 		model_indep.search_shotgun(self.params["search_iterations"])
		# 	# print("Finished Search Under Independent Model")

		# 	causal_set_indep = model_indep.get_causal_set(self.params["confidence"])
		# 	assert all([i == 0 or i == 1 for i in causal_set_indep])
		# 	causal_set_indep_size = sum(causal_set_indep)
		# 	result["set_sizes_indep"].append(causal_set_indep_size)
		# 	# print(causal_set_indep_size) ####
		# 	# print(model_eqtl.get_probs_sorted()) ####
		# 	# model_eqtl.get_probs_sorted() ####
		# 	# print(model_indep.get_probs()[tuple(causal_config)]) ####
		# 	# print(model_indep.get_probs()[null]) ####
		# 	x = model_indep.imbalance_stats
		# 	result["max_stat_ase_indep"] = abs(max(x.min(), x.max(), key=abs)) 

		# 	recall = self._recall(causal_set_indep, causal_config)
		# 	# for ind, val in enumerate(causal_config):
		# 	# 	if val == 1:
		# 	# 		if causal_set_indep[ind] != 1:
		# 	# 			recall = 0
		# 	result["recall_indep"].append(recall)
		# 	# print(recall) ####


		# 	# print("Initializing eQTL Model")
		# 	model_inputs_eqtl = model_inputs.copy()
		# 	model_inputs_eqtl.update({
		# 		"imbalance": np.zeros(shape=0), 
		# 		"phases": np.zeros(shape=(0,0)),
		# 		"imbalance_corr": np.zeros(shape=(0,0)),
		# 		"imbalance_errors": np.zeros(shape=0),
		# 		"imbalance_stats": np.zeros(shape=0),
		# 		"num_ppl_imbalance": 0,
		# 		"num_snps_imbalance": 0,
		# 		"corr_stats": 0.0,
		# 		"imbalance_var_prior": imbalance_var_prior,
		# 		"total_exp_var_prior": total_exp_var_prior,
		# 		"cross_corr_prior": 0.0,
		# 	})
		# 	# print("Finished Initializing eQTL Model")
		# 	# print("Starting Search Under eQTL Model")
		# 	model_eqtl = Finemap(**model_inputs_eqtl)
		# 	model_eqtl.initialize()
		# 	if self.params["search_mode"] == "exhaustive":
		# 		model_eqtl.search_exhaustive(self.params["min_causal"], self.params["max_causal"])
		# 	elif self.params["search_mode"] == "shotgun":
		# 		model_eqtl.search_shotgun(self.params["search_iterations"])
		# 	# print("Finished Search Under eQTL Model")

		# 	causal_set_eqtl = model_eqtl.get_causal_set(self.params["confidence"])
		# 	assert all([i == 0 or i == 1 for i in causal_set_eqtl])
		# 	causal_set_eqtl_size = sum(causal_set_eqtl)
		# 	result["set_sizes_eqtl"].append(causal_set_eqtl_size)
		# 	# print(causal_set_eqtl_size) ####
		# 	# print(model_eqtl.get_probs_sorted()) ####
		# 	# model_eqtl.get_probs_sorted() ####
		# 	# print(model_eqtl.get_probs()[tuple(causal_config)]) ####
		# 	# print(model_eqtl.get_probs()[tuple(null)]) ####
		# 	# ppas = model_eqtl.get_ppas()
		# 	# np.savetxt("ppas_eqtl.txt", np.array(ppas)) ####
		# 	# print(model_eqtl.total_exp_stats[causal_config == True][0]) ####
		# 	# eqtl_cstats.append(model_eqtl.total_exp_stats[causal_config == True][0]) ####

		# 	x = model_eqtl.imbalance_stats

		# 	recall = self._recall(causal_set_eqtl, causal_config)
		# 	# for ind, val in enumerate(causal_config):
		# 	# 	if val == 1:
		# 	# 		if causal_set_eqtl[ind] != 1:
		# 	# 			recall = 0
		# 	result["recall_eqtl"].append(recall)
		# 	# print(recall) ####

		# 	# print("Initializing ASE Model")
		# 	model_inputs_ase = model_inputs.copy()
		# 	model_inputs_ase.update({
		# 		"total_exp": np.zeros(shape=0), 
		# 		"genotypes_comb": np.zeros(shape=(0,0)),
		# 		"total_exp_corr": np.zeros(shape=(0,0)),
		# 		"total_exp_errors": np.zeros(shape=0),
		# 		"total_exp_stats": np.zeros(shape=0),
		# 		"num_ppl_total_exp": 0,
		# 		"num_snps_total_exp": 0,
		# 		"corr_stats": 0.0,
		# 		"imbalance_var_prior": imbalance_var_prior,
		# 		"total_exp_var_prior": total_exp_var_prior,
		# 		"cross_corr_prior": 0.0,
		# 	})
		# 	# print("Finished Initializing ASE Model")
		# 	# print("Starting Search Under ASE Model")
		# 	model_ase = Finemap(**model_inputs_ase)
		# 	model_ase.initialize()
		# 	if self.params["search_mode"] == "exhaustive":
		# 		model_ase.search_exhaustive(self.params["min_causal"], self.params["max_causal"])
		# 	elif self.params["search_mode"] == "shotgun":
		# 		model_ase.search_shotgun(self.params["search_iterations"])
		# 	# print("Finished Search Under ASE Model")

		# 	causal_set_ase = model_ase.get_causal_set(self.params["confidence"])
		# 	assert all([i == 0 or i == 1 for i in causal_set_ase])
		# 	causal_set_ase_size = sum(causal_set_ase)
		# 	result["set_sizes_ase"].append(causal_set_ase_size)
		# 	# print(causal_set_ase_size) ####
		# 	# print(model_eqtl.get_probs_sorted()) ####
		# 	# model_eqtl.get_probs_sorted() ####
		# 	# print(model_ase.get_probs()[tuple(causal_config)]) ####
		# 	# print(model_ase.get_probs()[tuple(null)]) ####
		# 	x = model_ase.imbalance_stats
		# 	result["max_stat_ase_ase"] = abs(max(x.min(), x.max(), key=abs)) 

		# 	recall = self._recall(causal_set_ase, causal_config)
		# 	# for ind, val in enumerate(causal_config):
		# 	# 	if val == 1:
		# 	# 		if causal_set_ase[ind] != 1:
		# 	# 			recall = 0
		# 	result["recall_ase"].append(recall)
		# 	# print(recall) ####

		# 	# if causal_set_ase_size == 181: ####
		# 	# 	ps = model_ase.get_probs_sorted()
		# 	# 	with open("null_ase.txt", "w") as null_ase:
		# 	# 		null_ase.write("\n".join("\t".join(str(j) for j in i) for i in ps))
		# 	# 	# np.savetxt("null_ase.txt", np.array(model_ase.get_probs_sorted()))
		# 	# 	raise Exception

		# 	model_caviar = EvalCaviar(
		# 		model_full, 
		# 		self.params["confidence"], 
		# 		self.params["max_causal"]
		# 	)
		# 	model_caviar.run()
		# 	causal_set_caviar = model_caviar.causal_set
		# 	causal_set_caviar_size = sum(causal_set_caviar)
		# 	result["set_sizes_caviar"].append(causal_set_caviar_size)
		# 	recall = self._recall(causal_set_caviar, causal_config)
		# 	result["recall_caviar"].append(recall)

		# 	model_caviar_ase = EvalCaviarASE(
		# 		model_full, 
		# 		self.params["confidence"], 
		# 		self.params["max_causal"]
		# 	)
		# 	model_caviar_ase.run()
		# 	causal_set_caviar_ase = model_caviar_ase.causal_set
		# 	causal_set_caviar_size_ase = sum(causal_set_caviar_ase)
		# 	result["set_sizes_caviar_ase"].append(causal_set_caviar_size_ase)
		# 	recall = self._recall(causal_set_caviar_ase, causal_config)
		# 	result["recall_caviar_ase"].append(recall)


		# print(np.std(eqtl_cstats)) ####
		# print(result["inclusions_caviar"]) ####
		
		print("Writing Result")
		self.primary_var_vals.append(self.params[self.params["primary_var"]])
		if "full" in model_flavors:
			result["recall_rate_full"] = np.mean(result["recall_full"])
			result["inclusion_rate_full"] = np.mean(result["inclusions_full"], axis=0)
		if "indep" in model_flavors:
			result["recall_rate_indep"] = np.mean(result["recall_indep"])
			result["inclusion_rate_indep"] = np.mean(result["inclusions_indep"], axis=0)
		if "eqtl" in model_flavors:
			result["recall_rate_eqtl"] = np.mean(result["recall_eqtl"])
			result["inclusion_rate_eqtl"] = np.mean(result["inclusions_eqtl"], axis=0)
		if "ase" in model_flavors:
			result["recall_rate_ase"] = np.mean(result["recall_ase"])
			result["inclusion_rate_ase"] = np.mean(result["inclusions_ase"], axis=0)
		if "cav" in model_flavors:
			result["recall_rate_caviar"] = np.mean(result["recall_caviar"])
			result["inclusion_rate_caviar"] = np.mean(result["inclusions_caviar"], axis=0)
		if "acav" in model_flavors:
			result["recall_rate_caviar_ase"] = np.mean(result["recall_caviar_ase"])
			result["inclusion_rate_caviar_ase"] = np.mean(result["inclusions_caviar_ase"], axis=0)
		
		test_path = self.set_output_folder()
		self.output_result(result, test_path)
		self.results.append(result)
		print("Finished Writing Result")



class Benchmark2d(Benchmark):
	def __init__(self, params):
		self.secondary_var_vals = []
		super(Benchmark2d, self).__init__(params)

	def set_output_folder(self):
		count_str = str(self.counter + 1).zfill(self.count_digits)
		test_folder = "{0}_{1}_{2}_{3}_{4}".format(
			count_str, 
			self.params["primary_var"], 
			str(self.params[self.params["primary_var"]]),
			self.params["secondary_var"], 
			str(self.params[self.params["secondary_var"]])
		)
		test_path = os.path.join(self.output_path, test_folder)
		os.makedirs(test_path)
		self.counter += 1
		return test_path

	@staticmethod
	def plot_heatmap(
		primary, 
		pname, 
		secondary, 
		sname, 
		vals,
		vname, 
		title_base, 
		output_path, 
		output_name_base,
		model_flavors
	):
		sns.set(font="Roboto")

		if True in vals:
			val = vals[True]
			df_full = pd.DataFrame({
				sname: secondary,
				pname: primary,
				vname: val,
			}).pivot(
				sname,
				pname,
				vname
			)

			title_full = title_base
			output_name_full = output_name_base + ".svg"
			sns.heatmap(df_full, annot=True, fmt=".1f", square=True)
			plt.title(title_full)
			plt.savefig(os.path.join(output_path, output_name_full))
			plt.clf()

		if "full" in model_flavors:
			vals_full = vals["full"]
			df_full = pd.DataFrame({
				sname: secondary,
				pname: primary,
				vname: vals_full,
			}).pivot(
				sname,
				pname,
				vname
			)

			title_full = title_base + " (Joint-Correlated)"
			output_name_full = output_name_base + "_full.svg"
			sns.heatmap(df_full, annot=True, fmt=".1f", square=True)
			plt.title(title_full)
			plt.savefig(os.path.join(output_path, output_name_full))
			plt.clf()

		if "indep" in model_flavors:
			vals_indep = vals["indep"]
			df_indep = pd.DataFrame({
				sname: secondary,
				pname: primary,
				vname: vals_indep,
			}).pivot(
				sname,
				pname,
				vname
			)

			title_indep = title_base + " (Joint-Independent)"
			output_name_indep = output_name_base + "_indep.svg"
			sns.heatmap(df_indep, annot=True, fmt=".1f", square=True)
			plt.title(title_indep)
			plt.savefig(os.path.join(output_path, output_name_indep))
			plt.clf()

		if "eqtl" in model_flavors:
			vals_eqtl = vals["eqtl"]
			df_eqtl = pd.DataFrame({
				sname: secondary,
				pname: primary,
				vname: vals_eqtl,
			}).pivot(
				sname,
				pname,
				vname
			)

			title_eqtl = title_base + " (eQTL-Only)"
			output_name_eqtl = output_name_base + "_eqtl.svg"
			sns.heatmap(df_eqtl, annot=True, fmt=".1f", square=True)
			plt.title(title_eqtl)
			plt.savefig(os.path.join(output_path, output_name_eqtl))
			plt.clf()

		if "ase" in model_flavors:
			vals_ase = vals["ase"]
			df_ase = pd.DataFrame({
				sname: secondary,
				pname: primary,
				vname: vals_ase,
			}).pivot(
				sname,
				pname,
				vname
			)

			title_ase = title_base + " (ASE-Only)"
			output_name_ase = output_name_base + "_ase.svg"
			sns.heatmap(df_ase, annot=True, fmt=".1f", square=True)
			plt.title(title_ase)
			plt.savefig(os.path.join(output_path, output_name_ase))
			plt.clf()

		if "cav" in model_flavors:
			vals_caviar = vals["cav"]
			df_caviar = pd.DataFrame({
				sname: secondary,
				pname: primary,
				vname: vals_caviar,
			}).pivot(
				sname,
				pname,
				vname
			)

			title_caviar = title_base + " (CAVIAR)"
			output_name_caviar = output_name_base + "_caviar.svg"
			sns.heatmap(df_caviar, annot=True, fmt=".1f", square=True)
			plt.title(title_caviar)
			plt.savefig(os.path.join(output_path, output_name_caviar))
			plt.clf()

		if "acav" in model_flavors:
			vals_caviar_ase = vals["acav"]
			df_caviar_ase = pd.DataFrame({
				sname: secondary,
				pname: primary,
				vname: vals_caviar_ase,
			}).pivot(
				sname,
				pname,
				vname
			)

			title_caviar_ase = title_base + " (CAVIAR-ASE)"
			output_name_caviar_ase = output_name_base + "_caviar_ase.svg"
			sns.heatmap(df_caviar_ase, annot=True, fmt=".1f", square=True)
			plt.title(title_caviar_ase)
			plt.savefig(os.path.join(output_path, output_name_caviar_ase))
			plt.clf()
	
	
	def output_summary(self):
		if self.params["model_flavors"] == "all":
			model_flavors = set(["full", "indep", "eqtl", "ase", "cav", "acav"])
		else:
			model_flavors = self.params["model_flavors"]
		num_trials = self.test_count
		num_trials_primary = self.params["test_count_primary"]
		num_trials_secondary = self.params["test_count_secondary"]
		results2d = [[None for _ in xrange(num_trials_primary)] for _ in xrange(num_trials_secondary)]
		for x in xrange(num_trials):
			row = x // num_trials_primary
			col = x % num_trials_secondary
			results2d[row][col] = self.results[x]

		means = {}
		recall_rate = {}

		if "full" in model_flavors:
			means["full"] = [np.mean(i["set_sizes_full"]) for i in self.results]
			recall_rate["full"] = [i["recall_rate_full"] for i in self.results]
		if "indep" in model_flavors:
			means["indep"] = [np.mean(i["set_sizes_indep"]) for i in self.results]
			recall_rate["indep"] = [i["recall_rate_indep"] for i in self.results]
		if "eqtl" in model_flavors:
			means["eqtl"] = [np.mean(i["set_sizes_eqtl"]) for i in self.results]
			recall_rate["eqtl"] = [i["recall_rate_eqtl"] for i in self.results]
		if "ase" in model_flavors:
			means["ase"] = [np.mean(i["set_sizes_ase"]) for i in self.results]
			recall_rate_ase = [i["recall_rate_ase"] for i in self.results]
		if "cav" in model_flavors:
			means["cav"] = [np.mean(i["set_sizes_caviar"]) for i in self.results]
			recall_rate["cav"] = [i["recall_rate_caviar"] for i in self.results]
		if "acav" in model_flavors:
			means["acav"] = [np.mean(i["set_sizes_caviar_ase"]) for i in self.results]
			recall_rate["acav"] = [i["recall_rate_caviar_ase"] for i in self.results]

		max_stat_ase = {True: [np.nanmean(i["max_stat_ase_full"])for i in self.results]}
		# max_stat_ase_indep = [np.mean(i["max_stat_ase_indep"]) for i in self.results]
		# max_stat_ase_ase = [np.mean(i["max_stat_ase_ase"]) for i in self.results]

		# print(max_stat_ase_full) ####

		# secondary = self.secondary_var_vals * num_trials_primary
		# primary =[i for i in self.primary_var_vals for _ in xrange(num_trials_secondary)]

		secondary = self.secondary_var_vals
		primary = self.primary_var_vals

		# print(self.secondary_var_vals) ####
		# print(self.primary_var_vals)
		# print(secondary) ####
		# print(primary) ####
		# print(means_full) ####

		self.plot_heatmap(
			primary,
			self.params["primary_var_display"],
			secondary,
			self.params["secondary_var_display"],
			means,
			"Mean Causal Set Size",
			"Mean Causal Set Sizes",
			self.output_path,
			"causal_sets",
			model_flavors
		)

		self.plot_heatmap(
			primary,
			self.params["primary_var_display"],
			secondary,
			self.params["secondary_var_display"],
			recall_rate,
			"Recall Rate",
			"Recall Rates",
			self.output_path,
			"recall",
			model_flavors
		)

		self.plot_heatmap(
			primary,
			self.params["primary_var_display"],
			secondary,
			self.params["secondary_var_display"],
			max_stat_ase,
			"Max Association Stat",
			"Max Association Statistics",
			self.output_path,
			"max_stat_ase",
			model_flavors
		)


		# df_full = pd.DataFrame({
		# 	self.params["secondary_var_display"]: secondary,
		# 	self.params["primary_var_display"]: primary,
		# 	"Mean Causal Set Size": means_full,
		# }).pivot(
		# 	self.params["secondary_var_display"],
		# 	self.params["primary_var_display"],
		# 	"Mean Causal Set Size"
		# )
		# df_indep = pd.DataFrame({
		# 	self.params["secondary_var_display"]: secondary,
		# 	self.params["primary_var_display"]: primary,
		# 	"Mean Causal Set Size": means_indep,
		# }).pivot(
		# 	self.params["secondary_var_display"],
		# 	self.params["primary_var_display"],
		# 	"Mean Causal Set Size"
		# )
		# df_eqtl = pd.DataFrame({
		# 	self.params["secondary_var_display"]: secondary,
		# 	self.params["primary_var_display"]: primary,
		# 	"Mean Causal Set Size": means_eqtl,
		# }).pivot(
		# 	self.params["secondary_var_display"],
		# 	self.params["primary_var_display"],
		# 	"Mean Causal Set Size"
		# )
		# df_ase = pd.DataFrame({
		# 	self.params["secondary_var_display"]: secondary,
		# 	self.params["primary_var_display"]: primary,
		# 	"Mean Causal Set Size": means_ase,
		# }).pivot(
		# 	self.params["secondary_var_display"],
		# 	self.params["primary_var_display"],
		# 	"Mean Causal Set Size"
		# )

		# df_full_recall = pd.DataFrame({
		# 	self.params["secondary_var_display"]: secondary,
		# 	self.params["primary_var_display"]: primary,
		# 	"Recall Rate": recall_rate_full,
		# }).pivot(
		# 	self.params["secondary_var_display"],
		# 	self.params["primary_var_display"],
		# 	"Recall Rate"
		# )
		# df_indep_recall = pd.DataFrame({
		# 	self.params["secondary_var_display"]: secondary,
		# 	self.params["primary_var_display"]: primary,
		# 	"Recall Rate": recall_rate_indep,
		# }).pivot(
		# 	self.params["secondary_var_display"],
		# 	self.params["primary_var_display"],
		# 	"Recall Rate"
		# )
		# df_eqtl_recall = pd.DataFrame({
		# 	self.params["secondary_var_display"]: secondary,
		# 	self.params["primary_var_display"]: primary,
		# 	"Recall Rate": recall_rate_eqtl,
		# }).pivot(
		# 	self.params["secondary_var_display"],
		# 	self.params["primary_var_display"],
		# 	"Recall Rate"
		# )
		# df_ase_recall = pd.DataFrame({
		# 	self.params["secondary_var_display"]: secondary,
		# 	self.params["primary_var_display"]: primary,
		# 	"Recall Rate": recall_rate_ase,
		# }).pivot(
		# 	self.params["secondary_var_display"],
		# 	self.params["primary_var_display"],
		# 	"Recall Rate"
		# )

		# sns.set()

		# sns.heatmap(df_full, annot=True, linewidths=1)
		# plt.title("Mean Causal Set Sizes (Full Model)")
		# plt.savefig(os.path.join(self.output_path, "causal_sets_full.svg"))
		# plt.clf()

		# sns.heatmap(df_indep, annot=True, linewidths=1)
		# plt.title("Mean Causal Set Sizes (Independent Likelihoods)")
		# plt.savefig(os.path.join(self.output_path, "causal_sets_indep.svg"))
		# plt.clf()

		# sns.heatmap(df_eqtl, annot=True, linewidths=1)
		# plt.title("Mean Causal Set Sizes (eQTL-Only)")
		# plt.savefig(os.path.join(self.output_path, "causal_sets_eqtl.svg"))
		# plt.clf()

		# sns.heatmap(df_ase, annot=True, linewidths=1)
		# plt.title("Mean Causal Set Sizes (ASE-Only)")
		# plt.savefig(os.path.join(self.output_path, "causal_sets_ase.svg"))
		# plt.clf()

		# sns.heatmap(df_full_recall, annot=True, linewidths=1)
		# plt.title("Recall Rate (Full Model)")
		# plt.savefig(os.path.join(self.output_path, "recall_full.svg"))
		# plt.clf()

		# sns.heatmap(df_indep_recall, annot=True, linewidths=1)
		# plt.title("Recall Rate (Independent Likelihoods)")
		# plt.savefig(os.path.join(self.output_path, "recall_indep.svg"))
		# plt.clf()

		# sns.heatmap(df_eqtl_recall, annot=True, linewidths=1)
		# plt.title("Recall Rate (eQTL-Only)")
		# plt.savefig(os.path.join(self.output_path, "recall_eqtl.svg"))
		# plt.clf()

		# sns.heatmap(df_ase_recall, annot=True, linewidths=1)
		# plt.title("Recall Rate (ASE-Only)")
		# plt.savefig(os.path.join(self.output_path, "recall_ase.svg"))
		# plt.clf()

	def test(self, **kwargs):
		super(Benchmark2d, self).test(**kwargs)
		self.secondary_var_vals.append(self.params[self.params["secondary_var"]])