from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

from . import Finemap
from . import SimAse
from . import Haplotypes

class Benchmark(object):
	res_path = os.path.join("results")
	def __init__(self, params):
		self.params = params
		self.haplotypes = Haplotypes()

		self.update_model_params()
		self.update_sim_params()

		self.time = datetime.now()
		self.timestamp = self.time.strftime("%y%m%d%H%M%f")
		self.counter = 0
		self.test_count = self.params["test_count"]
		self.count_digits = len(str(self.test_count))

		self.output_folder = "_" + self.params["test_name"]
		self.output_path = os.path.join(res_path, self.output_folder)

		self.results = []
		self.simulation = SimAse(self)
	
	def update_model_params(self):
		self.model_params = {
			"num_snps_imbalance": self.params["num_snps"],
			"num_snps_total_exp": self.params["num_snps"],
			"num_ppl_imbalance": self.params["num_ppl"],
			"num_ppl_total_exp": self.params["num_ppl"],
			"causal_status_prior": self.params["causal_status_prior"],
			"imbalance_var_prior": self.params["imbalance_var_prior"],
			"total_exp_var_prior": self.params["total_exp_var_prior"],
			"cross_corr_prior": self.params["cross_corr_prior"],
			"overdispersion": self.params["overdispersion"]
		}

	def update_sim_params(self):
		self.sim_params = {
			"num_snps": self.params["num_snps"],
			"num_ppl": self.params["num_ppl"],
			"var_effect_size": self.params["var_effect_size"],
			"overdispersion": self.params["overdispersion"],
			"prop_noise": self.params["prop_noise"],
			"baseline_exp": self.params["baseline_exp"]
			"num_causal": self.params["num_causal"]
			"ase_read_prop": self.params["ase_read_prop"]
			"overdispersion": self.params["overdispersion"]
		}
		self.simulation.update()

	@staticmethod
	def output_result(result, out_dir, params):
		set_sizes_full = result["set_sizes_full"]
		set_sizes_eqtl = result["set_sizes_eqtl"]
		recall_rate_full = result["recall_rate_full"]
		recall_rate_eqtl = result["recall_rate_eqtl"]

		params_str = "\n".join("{:<20}{:>20}".format(k, v) for k, v in params.viewitems())
		with open(os.path.join(out_dir, "parameters.txt"), "w") as params_file:
			params_file.write(params_str)

		with open(os.path.join(out_dir, "causal_set_sizes.txt"), "w") as cssfull:
			cssfull.write("\n".join(str(i) for i in set_sizes_full))

		with open(os.path.join(out_dir, "causal_set_sizes_eqtl_only.txt"), "w") as csseqtl:
			csseqtl.write("\n".join(str(i) for i in set_sizes_eqtl))

		with open(os.path.join(out_dir, "recall_rates.txt"), "w") as rrfull:
			rrfull.write("\n".join(str(i) for i in recall_rate_full))
		
		with open(os.path.join(out_dir, "recall_rates_eqtl_only.txt"), "w") as rreqtl:
			rreqtl.write("\n".join(str(i) for i in recall_rate_eqtl))


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
			"set_sizes_eqtl": [],
			"recall_rate_full": [],
			"recall_rate_eqtl": []
		}

		for _ in xrange(self.params["iterations"]):
			self.simulation.generate_data()
			sim_result = {
				"counts_A": self.simulation.counts_A,
				"counts_B": self.simulation.counts_B,
				"total_exp": self.simulation.total_exp,
				"hap_A": self.simulation.hap_A,
				"hap_B": self.simulation.hap_B
			}
			causal_config = self.simulation.causal_config()

			model_inputs = copy(self.model_params).update(sim_result)
			model_full = Finemap(**model_inputs)
			if self.params["search_mode"] == "exhaustive":
				model_full.search_exhaustive(self.params["max_causal"])
			elif self.params["search_mode"] == "shotgun":
				model_full.search_shotgun(self.params["search_iterations"])

			causal_set = model_full.get_causal_set(params["confidence"])
			assert all([i == 0 or i == 1 for i in causal_set])
			causal_set_size = sum(causal_set)
			result["set_sizes_full"].append(causal_set_size)

			recall = 1
			for val, ind in enumerate(causal_config):
				if val == 1:
					if causal_set[ind] != 1:
						recall = 0
			result["recall_rate_full"].append(recall)

			model_inputs_eqtl = copy(model_inputs).update(
				{"imbalance": np.zeros(shape=0), "phases": np.zeros(shape=(0,0))}
			)
			model_eqtl = Finemap(**model_inputs_eqtl)
			if self.params["search_mode"] == "exhaustive":
				model_eqtl.search_exhaustive(self.params["max_causal"])
			elif self.params["search_mode"] == "shotgun":
				model_eqtl.search_shotgun(self.params["search_iterations"])

			causal_set_eqtl = model_eqtl.get_causal_set(params["confidence"])
			assert all([i == 0 or i == 1 for i in causal_set_eqtl])
			causal_set_eqtl_size = sum(causal_set_eqtl)
			result["set_sizes_eqtl"].append(causal_set_eqtl_size)

			recall = 1
			for val, ind in enumerate(causal_config):
				if val == 1:
					if causal_set_eqtl[ind] != 1:
						recall = 0
			result["recall_rate_eqtl"].append(recall)

		self.output_result(result, test_path, self.params)
		self.results.append(result)



