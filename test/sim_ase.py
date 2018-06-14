from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import numpy.random as npr
import scipy.stats as sps
# import numpy.random.normal as normal
# import numpy.random.multivariate_normal as mvn
# import numpy.random.beta as betadist
# import numpy.random.binomial as binomial

class SimAse(object):
	def __init__(self, bm):
		self.bm = bm

	def update(self):
		self.num_snps = self.bm.sim_params["num_snps"]
		self.num_ppl = self.bm.sim_params["num_ppl"]
		self.var_effect_size = self.bm.sim_params["var_effect_size"]
		self.overdispersion = self.bm.sim_params["overdispersion"]
		self.exp_err_var = self.bm.sim_params["exp_err_var"]
		self.baseline_exp = self.bm.sim_params["baseline_exp"]
		self.num_causal = self.bm.sim_params["num_causal"]
		self.genotypes_A = self.bm.sim_params["genotypes_A"]
		self.genotypes_B = self.bm.sim_params["genotypes_B"]
		self.ase_read_prop = self.bm.sim_params["ase_read_prop"]
		self.overdispersion = self.bm.sim_params["overdispersion"]

		# self.num_snps = self.bm.num_snps
		# self.num_ppl = self.bm.num_ppl
		# self.var_effect_size = self.bm.var_effect_size
		# self.overdispersion = self.bm.overdispersion
		# self.exp_err_var = self.bm.exp_err_var
		# self.baseline_exp = self.bm.baseline_exp
		# # self.snp_prop_alt = self.bm.snp_prop_alt
		# self.num_causal = self.bm.num_causal
		# # self.corr_eigs = self.bm.corr_eigs
		# self.genotypes_A = self.bm.genotypes_A
		# self.genotypes_B = self.bm.genotypes_B
		# self.ase_read_prop = self.bm.ase_read_prop
		# self.overdispersion = self.bm.overdispersion

	# def _generate_correlations(self):
	# 	self.corrs = sps.random_correlation.rvs(self.corr_eigs)

	# def _generate_haplotype_single(self):
	# 	means = np.full(self.num_snps, self.snp_prop_alt)
	# 	hap_float = npr.multivariate_normal(means, self.corrs, self.num_ppl)
	# 	hap = np.where(hap_float >= 0.5, 1, 0) 
	# 	return hap

	# def _generate_genotypes(self):
	# 	self.genotypes_A = self._generate_haplotype_single()
	# 	self.genotypes_B = self._generate_haplotype_single()

	def _generate_effects(self):
		self.causal_effects = npr.normal(0, np.sqrt(self.var_effect_size), self.num_causal)
		causal_inds = npr.choice(self.num_snps, self.num_causal, replace=False)
		self.causal_snps = np.zeros(self.num_snps)
		np.put(self.causal_snps, causal_inds, self.causal_effects)

	def _generate_genotypes(self):
		self.genotypes_comb = self.hap_A + self.hap_B
		self.phases = self.hap_A - self.hap_B

	def _generate_expression(self):
		self.exp_A = self.hap_A.dot(self.causal_snps) + self.baseline_exp
		self.exp_B = self.hap_B.dot(self.causal_snps) + self.baseline_exp
		counts_A_ideal = np.exp(self.exp_A)
		counts_B_ideal = np.exp(self.exp_B)
		counts_total_ideal = counts_A_ideal + counts_B_ideal
		imbalance_ideal = self.exp_A - self.exp_B

		self.total_exp = npr.normal(np.log(counts_total_ideal), np.sqrt(self.exp_err_var))
		counts_total = np.exp(self.total_exp)
		ase_counts = npr.binomial(counts_total, self.ase_read_prop)
		betas = (1 / self.overdispersion - 1) * (1 / (1 + np.exp(imbalance_ideal)))
		alphas = (1 / self.overdispersion - 1) * (1 - 1 / (1 + np.exp(imbalance_ideal)))
		self.counts_A = npr.binomial(ase_counts, npr.beta(alphas, betas))
		self.counts_B = ase_counts - self.counts_A

	def generate_data(self):
		self._generate_effects()
		self._generate_genotypes()
		self._generate_expression()