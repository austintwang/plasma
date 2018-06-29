from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
from datetime import datetime
# import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from . import Finemap
from . import SimAse
from . import Haplotypes

class Benchmark(object):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	res_path = os.path.join("results")
	def __init__(self, params):
		self.params = params
		self.haplotypes = Haplotypes(params)

		self.results = []
		self.results_df = None
		self.primary_var_vals = []
		self.simulation = SimAse(self)

		self.update_model_params()
		self.update_sim_params()

		self.time = datetime.now()
		self.timestamp = self.time.strftime("%y%m%d%H%M%f")
		self.counter = 0
		self.test_count = self.params["test_count"]
		self.count_digits = len(str(self.test_count))

		self.output_folder = "_" + self.params["test_name"]
		self.output_path = os.path.join(self.dir_path, self.res_path, self.output_folder)

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
			"var_effect_size": self.params["var_effect_size"],
			"overdispersion": self.params["overdispersion"],
			"prop_noise_eqtl": self.params["prop_noise_eqtl"],
			"prop_noise_ase": self.params["prop_noise_ase"],
			"baseline_exp": self.params["baseline_exp"],
			"num_causal": self.params["num_causal"],
			"ase_read_prop": self.params["ase_read_prop"]
		}
		self.simulation.update()

	def output_result(self, result, out_dir):
		title_var = self.params["primary_var_display"]
		var_value = str(self.params[self.params["primary_var"]])

		set_sizes_full = result["set_sizes_full"]
		set_sizes_eqtl = result["set_sizes_eqtl"]
		recall_rate_full = result["recall_rate_full"]
		recall_rate_eqtl = result["recall_rate_eqtl"]

		params_str = "\n".join("{:<20}{:>20}".format(k, v) for k, v in self.params.viewitems())
		with open(os.path.join(out_dir, "parameters.txt"), "w") as params_file:
			params_file.write(params_str)

		with open(os.path.join(out_dir, "causal_set_sizes.txt"), "w") as cssfull:
			cssfull.write("\n".join(str(i) for i in set_sizes_full))

		with open(os.path.join(out_dir, "causal_set_sizes_eqtl_only.txt"), "w") as csseqtl:
			csseqtl.write("\n".join(str(i) for i in set_sizes_eqtl))

		with open(os.path.join(out_dir, "recall_rates.txt"), "w") as rr:
			rr.write("Full:{:>15}\neQTL-only:{:>15}".format(recall_rate_full, recall_rate_eqtl))

		sns.set(style="white")
		sns.distplot(
			set_sizes_full,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="Full"
		)
		sns.distplot(
			set_sizes_eqtl,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="eQTL-Only"
		)
		plt.legend(prop={"size": 16}, title="Model")
		plt.xlabel("Set Size")
		plt.ylabel("Density")
		plt.title("Distribution of Causal Set Sizes, {0} = {1}".format(title_var, var_value))
		plt.savefig(os.path.join(out_dir, "Set_size_distribution.svg"))
		plt.clf()		

	def output_summary(self):
		recall_full = [i["recall_rate_full"] for i in self.results]
		recall_eqtl = [i["recall_rate_eqtl"] for i in self.results]
		# sets_full = [i["set_sizes_full"] for i in self.results]
		# sets_eqtl = [i["set_sizes_eqtl"] for i in self.results]

		title_var = self.params["primary_var_display"]

		sns.set(style="white")
		sns.lmplot(self.primary_var_vals, recall_full)
		sns.lmplot(self.primary_var_vals, recall_eqtl)
		plt.legend(prop={"size": 16}, title="Model")
		plt.xlabel(title_var)
		plt.ylabel("Recall Rate")
		plt.title("Recall Rates Across {0}".format(title_var))
		plt.savefig(os.path.join(self.output_path, "recall_rates.svg"))
		plt.clf()

		dflst = []
		for ind, dct in enumerate(self.results):
			var_value = self.primary_var_vals[ind]
			for i in dct["set_sizes_full"]:
				dflst.append([i, var_value, "Full"])
			for i in dct["set_sizes_eqtl"]:
				dflst.append([i, var_value, "eQTL-Only"])
		res_df = pd.Dataframe(dflst, columns=["Set Size", title_var, "Model"])
		
		sns.set(style="whitegrid")
		sns.violinplot(
			x=title_var,
			y="Set Size",
			hue="Model",
			data=res_df,
			split=True,
			inner="quartile"
		)
		plt.title("Causal Set Sizes across {0}".format(title_var))
		plt.savefig(os.path.join(self.output_path, "causal_sets.svg"))
		plt.clf()

	def test(self, **kwargs):
		count_str = str(self.counter + 1).zfill(self.count_digits)
		test_folder = "{0}_{1}_{2}".format(
			count_str, 
			self.params["primary_var"], 
			str(self.params[self.params["primary_var"]])
		)
		test_path = os.path.join(self.output_path, test_folder)

		for k, v in kwargs.viewitems():
			self.params[k] = v
		self.update_model_params()
		self.update_sim_params()

		result = {
			"set_sizes_full": [],
			"set_sizes_indep": [],
			"set_sizes_eqtl": [],
			"set_sizes_ase": [],
			"recall_rate_full": [],
			"recall_rate_indep": [],
			"recall_rate_eqtl": [],
			"recall_rate_ase": [],
		}
		self.primary_var_vals.append(self.params[self.params["primary_var"]])

		for itr in xrange(self.params["iterations"]):
			print("\nIteration {0}".format(str(itr + 1)))
			print("Generating Simulation Data")
			self.simulation.generate_data()
			sim_result = {
				"counts_A": self.simulation.counts_A,
				"counts_B": self.simulation.counts_B,
				"total_exp": self.simulation.total_exp,
				"hap_A": self.simulation.hap_A,
				"hap_B": self.simulation.hap_B
			}
			causal_config = self.simulation.causal_config
			# print(causal_config) ####
			print("Finished Generating Simulation Data")

			# print(sim_result["hap_A"].tolist()) ####
			# print(sim_result["hap_B"].tolist()) ####
			null = tuple([0] * self.params["num_snps"]) ####

			print("Initializing Full Model")
			model_inputs = self.model_params.copy()
			model_inputs.update(sim_result)
			# print(model_inputs) ####
			model_full = Finemap(**model_inputs)
			model_full.initialize()
			print("Finished Initializing Full Model")
			print("Starting Search")
			if self.params["search_mode"] == "exhaustive":
				model_full.search_exhaustive(self.params["max_causal"])
			elif self.params["search_mode"] == "shotgun":
				model_full.search_shotgun(self.params["search_iterations"])
			print("Finished Search Under Full Model")

			causal_set = model_full.get_causal_set(self.params["confidence"])
			assert all([i == 0 or i == 1 for i in causal_set])
			causal_set_size = sum(causal_set)
			result["set_sizes_full"].append(causal_set_size)
			print(causal_set_size) ####
			print(model_full.get_probs()[tuple(causal_config)]) ####


			recall = 1
			for ind, val in enumerate(causal_config):
				if val == 1:
					if causal_set[ind] != 1:
						recall = 0
			result["recall_rate_full"].append(recall)
			print(recall) ####
			# print(model_full.get_probs_sorted()[:10]) ####


			print("Initializing Independent Model")
			model_inputs_indep = model_inputs.copy()
			model_inputs_indep.update({
				"cross_corr_prior": 0.0, 
				"corr_stats": 0.0,
			})
			print("Finished Initializing Independent Model")
			print("Starting Search Under Independent Model")
			model_indep = Finemap(**model_inputs_indep)
			model_indep.initialize()
			if self.params["search_mode"] == "exhaustive":
				model_indep.search_exhaustive(self.params["max_causal"])
			elif self.params["search_mode"] == "shotgun":
				model_indep.search_shotgun(self.params["search_iterations"])
			print("Finished Search Under Independent Model")

			causal_set_indep = model_indep.get_causal_set(self.params["confidence"])
			assert all([i == 0 or i == 1 for i in causal_set_indep])
			causal_set_indep_size = sum(causal_set_indep)
			result["set_sizes_indep"].append(causal_set_indep_size)
			print(causal_set_indep_size) ####
			# print(model_eqtl.get_probs_sorted()) ####
			# model_eqtl.get_probs_sorted() ####
			print(model_indep.get_probs()[tuple(causal_config)]) ####
			print(model_indep.get_probs()[null]) ####

			recall = 1
			for ind, val in enumerate(causal_config):
				if val == 1:
					if causal_set_indep[ind] != 1:
						recall = 0
			result["recall_rate_indep"].append(recall)
			print(recall) ####


			print("Initializing eQTL Model")
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
			})
			print("Finished Initializing eQTL Model")
			print("Starting Search Under eQTL Model")
			model_eqtl = Finemap(**model_inputs_eqtl)
			model_eqtl.initialize()
			if self.params["search_mode"] == "exhaustive":
				model_eqtl.search_exhaustive(self.params["max_causal"])
			elif self.params["search_mode"] == "shotgun":
				model_eqtl.search_shotgun(self.params["search_iterations"])
			print("Finished Search Under eQTL Model")

			causal_set_eqtl = model_eqtl.get_causal_set(self.params["confidence"])
			assert all([i == 0 or i == 1 for i in causal_set_eqtl])
			causal_set_eqtl_size = sum(causal_set_eqtl)
			result["set_sizes_eqtl"].append(causal_set_eqtl_size)
			print(causal_set_eqtl_size) ####
			# print(model_eqtl.get_probs_sorted()) ####
			# model_eqtl.get_probs_sorted() ####
			print(model_eqtl.get_probs()[tuple(causal_config)]) ####
			print(model_eqtl.get_probs()[tuple(null)]) ####

			recall = 1
			for ind, val in enumerate(causal_config):
				if val == 1:
					if causal_set_eqtl[ind] != 1:
						recall = 0
			result["recall_rate_eqtl"].append(recall)
			print(recall) ####

			print("Initializing ASE Model")
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
			})
			print("Finished Initializing ASE Model")
			print("Starting Search Under ASE Model")
			model_ase = Finemap(**model_inputs_ase)
			model_ase.initialize()
			if self.params["search_mode"] == "exhaustive":
				model_ase.search_exhaustive(self.params["max_causal"])
			elif self.params["search_mode"] == "shotgun":
				model_ase.search_shotgun(self.params["search_iterations"])
			print("Finished Search Under ASE Model")

			causal_set_ase = model_ase.get_causal_set(self.params["confidence"])
			assert all([i == 0 or i == 1 for i in causal_set_ase])
			causal_set_ase_size = sum(causal_set_ase)
			result["set_sizes_ase"].append(causal_set_ase_size)
			print(causal_set_ase_size) ####
			# print(model_eqtl.get_probs_sorted()) ####
			# model_eqtl.get_probs_sorted() ####
			print(model_ase.get_probs()[tuple(causal_config)]) ####
			print(model_ase.get_probs()[tuple(null)]) ####

			recall = 1
			for ind, val in enumerate(causal_config):
				if val == 1:
					if causal_set_ase[ind] != 1:
						recall = 0
			result["recall_rate_ase"].append(recall)
			print(recall) ####

		print("Writing Result")
		self.output_result(result, test_path)
		self.results.append(result)
		print("Finished Writing Result")



