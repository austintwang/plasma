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

# print("hhhh") ####
# from . import Haplotypes
# print("hhhhh") ####

class SimAse(object):
	def __init__(self, bm):
		self.bm = bm
		self.haplotypes = bm.haplotypes

	def update(self):
		self.num_snps = self.bm.sim_params["num_snps"]
		self.num_ppl = self.bm.sim_params["num_ppl"]
		self.var_effect_size = self.bm.sim_params["var_effect_size"]
		self.overdispersion = self.bm.sim_params["overdispersion"]
		self.prop_noise_eqtl = self.bm.sim_params["prop_noise_eqtl"]
		self.prop_noise_ase = self.bm.sim_params["prop_noise_ase"]
		self.baseline_exp = self.bm.sim_params["baseline_exp"]
		self.num_causal = self.bm.sim_params["num_causal"]
		# self.genotypes_A = self.bm.sim_params["genotypes_A"]
		# self.genotypes_B = self.bm.sim_params["genotypes_B"]
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
		self.causal_config = np.zeros(self.num_snps)
		np.put(self.causal_config, causal_inds, 1)
		self.causal_snps = np.zeros(self.num_snps)
		np.put(self.causal_snps, causal_inds, self.causal_effects)
		np.savetxt("causal_snps.txt", self.causal_snps) ####

	def _generate_genotypes(self):
		self.genotypes_comb = self.hap_A + self.hap_B
		self.phases = self.hap_A - self.hap_B

	# @staticmethod
	# def _draw_bb(ase_counts, alphas, betas):
	# 	counts = 0
	# 	while counts == 0:
	# 		counts = npr.binomial(ase_counts, npr.beta(alphas, betas))
	# 	return counts

	def _generate_expression(self):
		self.exp_A = self.hap_A.dot(self.causal_snps) + self.baseline_exp
		self.exp_B = self.hap_B.dot(self.causal_snps) + self.baseline_exp
		counts_A_ideal = np.exp(self.exp_A)
		counts_B_ideal = np.exp(self.exp_B)
		counts_total_ideal = counts_A_ideal + counts_B_ideal
		imbalance_ideal = self.exp_A - self.exp_B
		np.savetxt("imbalance_ideal.txt", imbalance_ideal) ####
		self.total_exp_ideal = np.log(counts_total_ideal)
		np.savetxt("total_exp_ideal.txt", self.total_exp_ideal) ####
		ideal_exp_var = np.var(self.total_exp_ideal)
		# total_var = ideal_exp_var * (self.prop_noise_eqtl / (1 + self.prop_noise_eqtl))
		noise_var = ideal_exp_var * (self.prop_noise_eqtl / (1 - self.prop_noise_eqtl))

		self.total_exp = npr.normal(self.total_exp_ideal, np.sqrt(noise_var))
		np.savetxt("total_exp.txt", self.total_exp) ####
		# print(self.total_exp_ideal) ####
		# print(self.total_exp) ####
		# smalls = np.nonzero(self.total_exp < 0.69)[0]
		# while smalls.size != 0:
		# 	small_subs = npr.normal(self.total_exp_ideal[smalls], np.sqrt(total_var))
		# 	np.put(
		# 		self.total_exp,
		# 		smalls,
		# 		small_subs
		# 	)

		counts_total = np.exp(self.total_exp).astype(int)
		# print(counts_total) ####
		# counts_total = 100 ####
		# print(self.ase_read_prop) ####
		
		ase_counts = npr.binomial(
			counts_total, 
			self.ase_read_prop * (1 - self.prop_noise_ase)
		)
		ase_counts = 100 ####

		trans_counts_exp = counts_total * self.ase_read_prop * self.prop_noise_ase
		trans_counts_A = npr.poisson(trans_counts_exp / 2) + 1
		trans_counts_B = npr.poisson(trans_counts_exp / 2) + 1
		trans_counts = trans_counts_A + trans_counts_B

		# lows = np.nonzero(ase_counts < 2)[0]
		# while lows.size != 0:
		# 	low_subs = npr.binomial(counts_total[lows], self.ase_read_prop)
		# 	np.put(
		# 		ase_counts,
		# 		lows,
		# 		low_subs
		# 	)
		# print(np.mean(counts_total * self.ase_read_prop)) ####
		betas = (1 / self.overdispersion - 1) * (1 / (1 + np.exp(imbalance_ideal)))
		alphas = (1 / self.overdispersion - 1) * (1 / (1 + np.exp(-imbalance_ideal)))
		# print(alphas) ####
		# print(betas) ####
		# print(self.overdispersion) ####
		@np.vectorize
		def _bb(counts, alpha, beta):
			p = npr.beta(alpha, beta, size=counts)
			# print(p) ####
			return np.sum(npr.binomial(1, p))

		# counts_A_ase = npr.binomial(ase_counts, npr.beta(alphas, betas))
		counts_A_ase = _bb(ase_counts, alphas, betas)

		self.counts_A = counts_A_ase + trans_counts_A
		
		# zeros = np.nonzero(self.counts_A == 0)[0]
		# while zeros.size != 0:
		# 	zero_subs = npr.binomial(ase_counts[zeros], npr.beta(alphas[zeros], betas[zeros]))
		# 	np.put(
		# 		self.counts_A, 
		# 		zeros, 
		# 		zero_subs
		# 	)
		# 	zeros = np.nonzero(self.counts_A == 0)[0]
		# 	print(zeros.size) ####
		# 	print(ase_counts[zeros]) ####
		# 	print(alphas[zeros]) ####
		# 	print(betas[zeros]) ####
		# 	print(zero_subs) ####
		
		# alls = np.nonzero(self.counts_A == ase_counts)[0]
		# while alls.size != 0:
		# 	all_subs = npr.binomial(ase_counts[alls], npr.beta(alphas[alls], betas[alls]))
		# 	np.put(
		# 		self.counts_A, 
		# 		alls, 
		# 		all_subs
		# 	)
		# 	alls = np.nonzero(self.counts_A == ase_counts)[0]
		# 	print(alls.size) ####
		# 	print(ase_counts[alls])
		# 	print(alphas[alls]) ####
		# 	print(betas[alls]) ####
		# 	print(all_subs) ####

		self.counts_B = ase_counts - counts_A_ase + trans_counts_B
		np.savetxt("counts_A.txt", self.counts_A) ####
		np.savetxt("counts_B.txt", self.counts_B) ####
		# print(self.counts_A) ####

	def generate_data(self):
		self.hap_A, self.hap_B = self.haplotypes.draw_haps()
		self._generate_effects()
		self._generate_genotypes()
		self._generate_expression()

		alt_counts = (
			(self.hap_A.T * (1 - self.hap_B.T) * self.counts_A).sum(1) 
			+ (self.hap_B.T * (1 - self.hap_A.T) * self.counts_B).sum(1) ####
		)
		wt_counts = (
			(self.hap_B.T * (1 - self.hap_A.T) * self.counts_A).sum(1) 
			+ (self.hap_A.T * (1 - self.hap_B.T) * self.counts_B).sum(1) ####
		)
		totals = alt_counts + wt_counts ####
		# print(alt_counts) ####
		# print(totals) ####
		# print(self.counts_A * (1 - self.counts_B)) ####
		test = np.array([sps.binom_test(alt_counts[i], n=totals[i]) for i in xrange(self.num_snps)]) ####
		np.savetxt("alt_counts.txt", alt_counts) ####
		np.savetxt("binom_test.txt", test) ####