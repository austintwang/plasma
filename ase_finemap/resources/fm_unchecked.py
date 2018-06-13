from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np 
import scipy.linalg.lapack as lp
import itertools

from .evaluator import Evaluator

class FmUnchecked(object):
	IMBALANCE_VAR_PRIOR_DEFAULT = 0.1
	TOTAL_EXP_VAR_PRIOR_DEFAULT = 0.1
	CROSS_CORR_PRIOR_DEFAULT = 1.0

	def __init__(self, **kwargs):
		self.num_snps_imbalance = kwargs.get("num_snps_imbalance", None)
		self.num_snps_total_exp = kwarg.get("num_snps_total_exp", None)
		self.num_ppl_imbalance = kwargs.get("num_ppl_imbalance", None)
		self.num_ppl_total_exp = kwargs.get("num_ppl_total_exp", None)

		self.causal_status_prior = kwargs.get("causal_status_prior", None)

		self.imbalance_var_prior = kwargs.get("imbalance_var_prior", IMBALANCE_VAR_PRIOR_DEFAULT)
		self.total_exp_var_prior = kwargs.get("total_exp_var_prior", TOTAL_EXP_VAR_PRIOR_DEFAULT)
		self.cross_corr_prior = kwargs.get("cross_corr_prior", CROSS_CORR_PRIOR_DEFAULT)

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
		self.std_error = kwargs.get("std_error", None)

		self.exp_A = kwargs.get("exp_A", None)
		self.exp_B = kwargs.get("exp_B", None)

		self.genotypes_A = kwargs.get("genotypes_A", None)
		self.genotypes_B = kwargs.get("genotypes_B", None)

		self._beta = None
		self._mean = None

		self._covdiag_phi = None
		self._covdiag_beta = None

		self.evaluator = None

	def _calc_causal_status_prior(self):
		if self.causal_status_prior != None:
			return

		self.causal_status_prior = 1.0 / max(self.num_snps_imbalance, self.num_snps_total_exp)

	def _calc_imbalance(self):
		if self.imbalance != None:
			return

		self.imbalance = np.log(self.exp_A) - np.log(self.exp_B)
	
	def _calc_phases(self):
		if self.phases != None:
			return

		self.phases = self.genotypes_A - self.genotypes_B

	def _calc_total_exp(self):
		if self.total_exp != None:
			return

		self.total_exp = np.log(self.exp_A) + np.log(self.exp_B)

	def _calc_genotypes_comb(self):
		if self.genotypes_comb != None:
			return

		self.genotypes_comb = self.genotypes_A + self.genotypes_B

	def _calc_imbalance_errors(self):
		if self.imbalance_errors != None:
			return

		self._calc_imbalance()

		counts = self.exp_A + self.exp_B
		self.imbalance_errors = (
			2 / counts
			* (1 + np.cosh(self.imbalance)) 
			* (1 + self.overdispersion * (counts - 1))
		)

	def _calc_imbalance_stats(self):
		if self.imbalance_stats != None:
			return

		self._calc_imbalance_errors()
		self._calc_phases()
		self._calc_imbalance()

		phases = self.phases
		phasesT = phases.T
		weights = 1 / self.imbalance_errors
		denominator = np.empty(self.num_ppl_imbalance)
		for ph, ind in enumerate(phases):
			denominator[ind] = ph.dot(weights).dot(ph)
		phi = denominator * phasesT.dot(weights.dot(self.imbalance))
		varphi = denominator * denominator * np.sum((phasesT * phasesT), axis=1)
		self.imbalance_stats = phi / varphi

	def _calc_imbalance_corr(self):
		if self.imbalance_corr != None:
			return

		self._calc_imbalance_errors()
		self._calc_phases()

		imbalance_errors = self.imbalance_errors
		phases = self.phases
		phasesT = phases.T
		weight_matrix = np.diag(1 / imbalance_errors)
		cov = phasesT.dot(weight_matrix).dot(phases) / self.num_ppl_imbalance
		covdiag = np.diag(cov)
		self._covdiag_phi = covdiag
		denominator = np.sqrt(np.outer(covdiag, covdiag))
		self.imbalance_corr = cov / denominator

	def _calc_beta(self):
		self._calc_genotypes_comb()
		self._calc_total_exp()

		genotypes_comb = self.genotypes_comb
		genotypes_combT = genotypes_comb.T
		mean = np.sum(self.total_exp) / self.num_ppl_total_exp
		denominator = np.empty(self.num_ppl_total_exp)
		for ge, ind in enumerate(genotypes_comb):
			denominator[ind] = ge.dot(ge)
		self._beta = denominator * genotypes_combT.dot(self.total_exp - mean)
		self._mean = mean

	def _calc_total_exp_error(self):
		if self.std_error != None:
			return

		self._calc_beta()

		residuals = self.genotypes_comb.dot(self._beta) - self._mean
		self.std_error = residuals.dot(residuals) / (self.num_ppl_total_exp - 2)


	def _calc_total_exp_stats(self):
		if self.total_exp_stats != None:
			return

		self._calc_total_exp_errors()

		varbeta = denominator * denominator * np.sum((phasesT * phasesT), axis=1) * self.std_error
		self.total_exp_stats = self._beta / varbeta

	def _calc_total_exp_corr(self):
		if self.total_exp_stats != None:
			return

		self._calc_genotypes_comb()

		genotypes_comb = self.genotypes_comb
		genotypes_combT = genotypes_comb.T
		cov = (genotypes_combT.dot(genotypes_comb) 
			- np.outer(np.sum(genotypes_combT, axis=1)) / self.num_ppl_total_exp) / self.num_ppl_total_exp
		covdiag = np.diag(cov)
		self._covdiag_beta = covdiag
		denominator = np.sqrt(np.outer(covdiag, covdiag))
		self.total_exp_corr = cov / denominator

	def _calc_cross_corr(self):
		if self.cross_corr != None:
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
		self.evaluator.eval(configuration)
		for i in xrange(num_iterations):
			neighbors = []
			for val, ind in enumerate(configuration):
				# Add causal variant
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
				for val2, ind2 in enumerate(configuration, start=ind+1):
					if val2 != val:
						neighbor = configuration.copy()
						neighbor[ind] = val2
						neighbor[ind2] = val
						neighbors.append(neighbor)

			posteriors = []
			for n in neighbors:
				posteriors.append(self.evaluator.eval(n))
			posteriors = np.array(posteriors)

			dist = posteriors / np.sum(posteriors)
			configuration = np.random.choice(neighbors, p=dist)

	def get_probs(self):
		return self.evaluator.get_probs()

	def get_probs_sorted(self):
		return self.evaluator.get_probs_sorted()

	def get_ppas(self):
		return self.evaluator.get_ppas()

	def reset_mapping(self):
		self.evaluator.reset()