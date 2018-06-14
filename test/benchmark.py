from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np

from . import Finemap
from . import SimAse

class Benchmark(object):
	
	def __init__(self, params):
		self.params = params

		self.load_haplotypes()
		self.update_model_params()
		self.update_sim_params()

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
			"exp_err_var": self.params["exp_err_var"],
			"baseline_exp": self.params["baseline_exp"]
			"num_causal": self.params["num_causal"]
			"ase_read_prop": self.params["ase_read_prop"]
			"overdispersion": self.params["overdispersion"]
		}

	def load_haplotypes(self):
		pass

	def test(self, **kwargs):
		for k, v in kwargs.viewitems():
			self.params[k] = v
		self.update_model_params()
		self.update_sim_params()

		self.simulation = SimAse(self)
		self.simulation.generate_data()
		self.sim_result = {
			"counts_A": self.simulation.counts_A,
			"counts_B": self.simulation.counts_B,
			"total_exp": self.simulation.total_exp,
			"hap_A": self.simulation.hap_A,
			"hap_B": self.simulation.hap_B
		}
		model_inputs = self.model_params + self.sim_result
		model_result = []
		for _ in xrange(self.params["iterations"]):
			self.model = Finemap(**model_inputs)
			if self.params["search_mode"] == "exhaustive":
				self.model.search_exhaustive(self.params["max_causal"])
			elif self.params["search_mode"] == "shotgun":
				self.model.search_shotgun(self.params["search_iterations"])
			model_result.append(self.model.get_probs_sorted())

		result = {
			"model_params": self.model_params,
			"sim_params": self.sim_params,
			"sim_result": self.sim_result,
			"model_result": self.model_result,
		}

		self.results.append(result)




