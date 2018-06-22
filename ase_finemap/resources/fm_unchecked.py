from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np 
import scipy.linalg.lapack as lp
import itertools

from .evaluator import Evaluator

class FmUnchecked(object):
	IMBALANCE_VAR_PRIOR_DEFAULT = 10.0
	TOTAL_EXP_VAR_PRIOR_DEFAULT = 10.0
	CROSS_CORR_PRIOR_DEFAULT = 0.95

	def __init__(self, **kwargs):
		self.num_snps_imbalance = kwargs.get("num_snps_imbalance", None)
		self.num_snps_total_exp = kwargs.get("num_snps_total_exp", None)
		self.num_ppl_imbalance = kwargs.get("num_ppl_imbalance", None)
		self.num_ppl_total_exp = kwargs.get("num_ppl_total_exp", None)

		self.causal_status_prior = kwargs.get("causal_status_prior", None)

		self.imbalance_var_prior = kwargs.get("imbalance_var_prior", self.IMBALANCE_VAR_PRIOR_DEFAULT)
		self.total_exp_var_prior = kwargs.get("total_exp_var_prior", self.TOTAL_EXP_VAR_PRIOR_DEFAULT)
		self.cross_corr_prior = kwargs.get("cross_corr_prior", self.CROSS_CORR_PRIOR_DEFAULT)

		self.imbalance_stats = kwargs.get("imbalance_stats", None)
		self.imbalance_corr = kwargs.get("imbalance_corr", None)
		self.total_exp_stats = kwargs.get("total_exp_stats", None)
		self.total_exp_corr = kwargs.get("total_exp_corr", None)
		self.cross_corr = kwargs.get("cross_corr", None)

		self.imbalance = kwargs.get("imbalance", None)
		self.phases = kwargs.get("phases", None)
		self.total_exp = kwargs.get("total_exp", None)
		self.genotypes_comb = kwargs.get("genotypes_comb", None)

		self.overdispersion = kwargs.get("overdispersion", None)
		self.imbalance_errors = kwargs.get("imbalance_errors", None)
		self.exp_error_var = kwargs.get("exp_error_var", None)

		self.counts_A = kwargs.get("counts_A", None)
		self.counts_B = kwargs.get("counts_B", None)

		self.hap_A = kwargs.get("hap_A", None)
		self.hap_B = kwargs.get("hap_B", None)

		self._beta = None
		self._mean = None
		self._beta_normalizer = None

		self._covdiag_phi = None
		self._covdiag_beta = None

		self.evaluator = None

	def _calc_counts(self):
		pass

	def _calc_haps(self):
		pass
	
	def _calc_causal_status_prior(self):
		if self.causal_status_prior is not None:
			return

		self.causal_status_prior = 1.0 / max(self.num_snps_imbalance, self.num_snps_total_exp)

	def _calc_imbalance(self):
		if self.imbalance is not None:
			return

		self._calc_counts()

		self.imbalance = np.log(self.counts_A) - np.log(self.counts_B)
		# print(self.imbalance) ####
	
	def _calc_phases(self):
		if self.phases is not None:
			return

		self._calc_haps()

		self.phases = self.hap_A - self.hap_B

	def _calc_total_exp(self):
		if self.total_exp is not None:
			return

		self._calc_counts()

		self.total_exp = np.log(self.counts_A) + np.log(self.counts_B)

	def _calc_genotypes_comb(self):
		if self.genotypes_comb is not None:
			return

		self._calc_haps()

		self.genotypes_comb = self.hap_A + self.hap_B
		# print(self.genotypes_comb) ####

	def _calc_imbalance_errors(self):
		if self.imbalance_errors is not None:
			return

		self._calc_imbalance()

		counts = self.counts_A + self.counts_B
		self.imbalance_errors = (
			2 / counts
			* (1 + np.cosh(self.imbalance)) 
			* (1 + self.overdispersion * (counts - 1))
		)
		# print(self.imbalance_errors) ####

	def _calc_imbalance_stats(self):
		if self.imbalance_stats is not None:
			return

		self._calc_imbalance_errors()
		self._calc_phases()
		self._calc_imbalance()

		phases = self.phases
		phasesT = phases.T
		weights = 1 / self.imbalance_errors
		denominator = 1 / ((phasesT * weights) * phasesT).sum(1) 
		# print(((phasesT * weights) * phasesT).sum(1)) ####
		# print(denominator) ####
		# denominator = np.empty(self.num_ppl_imbalance)
		# for ph, ind in enumerate(phases):
		# 	denominator[ind] = ph.dot(weights).dot(ph)
		phi = denominator * np.matmul(phasesT, (weights * self.imbalance))
		varphi = denominator * denominator * (phasesT * phasesT).sum(1) 
		self.imbalance_stats = phi / varphi

	def _calc_imbalance_corr(self):
		if self.imbalance_corr is not None:
			return

		self._calc_imbalance_errors()
		self._calc_phases()

		imbalance_errors = self.imbalance_errors
		phases = self.phases
		phasesT = phases.T
		weight_matrix = np.diag(1 / imbalance_errors)
		cov = phasesT.dot(weight_matrix).dot(phases) / self.num_ppl_imbalance
		covdiag = np.diag(cov)
		# print(covdiag) ####
		self._covdiag_phi = covdiag
		denominator = np.sqrt(np.outer(covdiag, covdiag))
		self.imbalance_corr = cov / denominator
		self.imbalance_corr = np.nan_to_num(self.imbalance_corr)
		np.fill_diagonal(self.imbalance_corr, 1.0)

	def _calc_beta(self):
		if self._beta is not None:
			return

		self._calc_genotypes_comb()
		self._calc_total_exp()

		genotypes_comb = self.genotypes_comb
		genotypes_combT = genotypes_comb.T
		mean = np.sum(self.total_exp) / self.num_ppl_total_exp
		denominator = 1 / (genotypes_combT * genotypes_combT).sum(1)
		# print((genotypes_combT * genotypes_combT).sum(1)) ####
		# print(denominator) ####
		# denominator = np.empty(self.num_ppl_total_exp)
		# for ge, ind in enumerate(genotypes_comb):
		# 	denominator[ind] = ge.dot(ge)
		self._beta = denominator * genotypes_combT.dot(self.total_exp - mean)
		# print(genotypes_combT.dot(self.total_exp - mean)) ####
		# print(self._beta) ####
		# print(self._beta.shape) ####
		self._mean = mean
		self._beta_normalizer = denominator 

	def _calc_total_exp_error(self):
		if self.exp_error_var is not None:
			return

		self._calc_beta()

		residuals = self.genotypes_comb.dot(np.nan_to_num(self._beta)) - self._mean
		# print(self._mean) ####
		# print(residuals) ####
		self.exp_error_var = residuals.dot(residuals) / (self.num_ppl_total_exp - 2)


	def _calc_total_exp_stats(self):
		if self.total_exp_stats is not None:
			return

		self._calc_genotypes_comb()
		self._calc_beta()
		self._calc_total_exp_error()

		genotypes_combT = self.genotypes_comb.T
		denominator = self._beta_normalizer

		varbeta = denominator * denominator * (
			(genotypes_combT * genotypes_combT).sum(1) * self.exp_error_var
		)
		# print(self.exp_error_var) ####
		# print(varbeta) ####
		# print(self._beta) ####
		self.total_exp_stats = self._beta / varbeta
		# print(self.total_exp_stats) ####

	def _calc_total_exp_corr(self):
		if self.total_exp_corr is not None:
			return

		self._calc_genotypes_comb()

		genotypes_comb = self.genotypes_comb
		# print(self.genotypes_comb) ####
		genotypes_combT = genotypes_comb.T
		means = np.sum(genotypes_combT, axis=1)
		cov = (
			(genotypes_combT.dot(genotypes_comb) 
			- np.outer(means, means) / self.num_ppl_total_exp) / self.num_ppl_total_exp
		)
		covdiag = np.diag(cov)
		self._covdiag_beta = covdiag
		denominator = np.sqrt(np.outer(covdiag, covdiag))
		self.total_exp_corr = cov / denominator
		self.total_exp_corr = np.nan_to_num(self.total_exp_corr)
		np.fill_diagonal(self.total_exp_corr, 1.0)

	def _calc_cross_corr(self):
		if self.cross_corr is not None:
			return

		self._calc_imbalance_errors()
		self._calc_phases()
		self._calc_genotypes_comb()
		self._calc_imbalance_corr()
		self._calc_total_exp_corr()

		if self.num_ppl_imbalance < self.num_ppl_total_exp:
			diff = self.num_ppl_total_exp - self.num_ppl_imbalance
			num = self.num_ppl_imbalance
			imbalance_errors = np.concatenate(self.imbalance_errors, np.zeros(diff))
			genotypes_comb = self.genotypes_comb
			phases = np.concatenate(self.phases, np.zeros(diff, self.num_snps_imbalance))
		elif self.num_ppl_imbalance > self.num_ppl_total_exp:
			diff = self.num_ppl_imbalance - self.num_ppl_total_exp
			num = self.num_ppl_total_exp
			imbalance_errors = self.imbalance_errors
			genotypes_comb = np.concatenate(self.genotypes_comb, np.zeros(diff, self.num_snps_total_exp))
			phases = self.phases
		else:
			num = self.num_ppl_imbalance
			imbalance_errors = self.imbalance_errors
			genotypes_comb = self.genotypes_comb
			phases = self.phases

		half_weights = np.sqrt(np.diag(1 / imbalance_errors))
		ccov = genotypes_comb.T.dot(half_weights).dot(phases) / num
		denominator = np.sqrt(np.outer(self._covdiag_phi, self._covdiag_beta))
		self.cross_corr = ccov / denominator
		self.cross_corr = np.nan_to_num(self.cross_corr)
		# print(self.cross_corr) ####

	def initialize(self):
		self._calc_causal_status_prior()
		self._calc_imbalance_stats()
		self._calc_total_exp_stats()
		self._calc_imbalance_corr()
		self._calc_total_exp_corr()
		self._calc_cross_corr()

		self.evaluator = Evaluator(self)

	def search_exhaustive(self, max_causal):
		m = max(self.num_snps_imbalance, self.num_snps_total_exp)
		for k in xrange(max_causal):
			base = [0] * k + [1] * (m - k)
			for c in itertools.permutations(base):
				self.evaluator.eval(np.array(c))

	def search_shotgun(self, num_iterations):
		m = max(self.num_snps_imbalance, self.num_snps_total_exp)
		configuration = np.zeros(m)
		# print(m) ####
		self.evaluator.eval(configuration)
		# print([(i, j) for i, j in enumerate(configuration, start=999)]) ####
		for i in xrange(num_iterations):
			neighbors = []
			for ind in xrange(m):
				val = configuration[ind]
				# Add causal variant
				# print(val, ind) ####
				if val == 0:
					neighbor = configuration.copy()
					neighbor[ind] = 1
					neighbors.append(neighbor)
				# Remove causal variant
				elif val == 1:
					neighbor = configuration.copy()
					neighbor[ind] = 0
					neighbors.append(neighbor)
				# Swap status with other variants
				for ind2 in xrange(ind+1, m):
					val2 = configuration[ind2]
					# if ind2 == 1000:
					# 	print(val2) ####
					# 	print() ####
					if val2 != val:
						neighbor = configuration.copy()
						neighbor[ind] = val2
						neighbor[ind2] = val
						neighbors.append(neighbor)

			posteriors = []
			for n in neighbors:
				posteriors.append(self.evaluator.eval(n))
			posteriors = np.array(posteriors)
			# print(posteriors) ####
			dist = posteriors / np.sum(posteriors)
			# print(neighbors) ####
			# print(dist) ####
			selection = np.random.choice(np.arange(len(neighbors)), p=dist)
			configuration = neighbors[selection]
			# print(configuration.shape) ####
			# print(configuration) ####
			# if i % 1 == 0: ####
			# 	print(i) ####

	def get_probs(self):
		return self.evaluator.get_probs()

	def get_probs_sorted(self):
		return self.evaluator.get_probs_sorted()

	def get_causal_set(self):
		return self.evaluator.get_causal_set()

	def get_ppas(self):
		return self.evaluator.get_ppas()

	def reset_mapping(self):
		self.evaluator.reset()