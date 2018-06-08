from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np 
import scipy.linalg.lapack as lp
import itertools

from .evaluator import Evaluator

class FM_unchecked(object):
	IMBALANCE_VAR_PRIOR_DEFAULT = 0.1
	TOTAL_EXP_VAR_PRIOR_DEFAULT = 0.1
	CROSS_CORR_PRIOR_DEFAULT = 0.8

	def __init__(self, **kwargs):
		self.num_snps = kwargs.get("num_snps", -1)
		self.num_ppl = kwargs.get("num_ppl", -1)

		self.causal_status_prior = kwargs.get("causal_status_prior", 1.0 / self.num_snps)

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
		self.exp_B = kwargs.get("genotypes_B", None)

		self._covdiag_phi = None
		self._covdiag_beta = None

		self.evaluator = None

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
		self._calc_total_exp()

		self.imbalance_errors = (
			2 / self.total_exp 
			* (1 + np.cosh(self.imbalance)) 
			* (1 + self.overdispersion * (self.total_exp - 1))
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
		denominator = np.empty(self.num_ppl)
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
		cov = phasesT.dot(weight_matrix).dot(phases)
		covdiag = np.diag(cov)
		self._covdiag_phi = covdiag
		denominator = np.sqrt(np.outer(covdiag, covdiag))
		self.imbalance_corr = cov / denominator

	def _calc_total_exp_stats(self):
		if self.total_exp_stats != None:
			return

		self._calc_genotypes_comb()
		self._calc_total_exp()

		genotypes_comb = self.genotypes_comb
		genotypes_combT = genotypes_comb.T
		mean = np.sum(self.total_exp) / self.num_ppl
		denominator = np.empty(self.num_ppl)
		for ge, ind in enumerate(genotypes_comb):
			denominator[ind] = ge.dot(ge)
		beta = denominator * genotypes_combT.dot(self.total_exp - mean)
		varbeta = denominator * denominator * np.sum((phasesT * phasesT), axis=1) * self.std_error
		self.total_exp_stats = beta / varbeta

	def _calc_total_exp_corr(self):
		if self.total_exp_stats != None:
			return

		self._calc_genotypes_comb()

		genotypes_comb = self.genotypes_comb
		genotypes_combT = genotypes_comb.T
		cov = genotypes_combT.dot(genotypes_comb) - np.outer(np.sum(genotypes_combT, axis=1)) / self.num_ppl
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

		imbalance_errors = self.imbalance_errors
		half_weights = np.sqrt(np.diag(1 / imbalance_errors))
		ccov = genotypes_combT.dot(half_weights).dot(phases)
		denominator = np.sqrt(np.outer(self._covdiag_phi, self._covdiag_beta))
		self.cross_corr = ccov / denominator

	def initialize(self):
		self._calc_imbalance_stats()
		self._calc_total_exp_stats()
		self._calc_imbalance_corr()
		self._calc_total_exp_corr()
		self._calc_cross_corr()

		self.evaluator = Evaluator(self)

	def search_exhaustive(self, max_causal):
		m = self.evaluator.num_snps
		for k in xrange(max_causal):
			base = [0] * k + [1] * (m - k)
			for c in itertools.permutations(base):
				self.evaluator.eval(np.array(c))

	def search_shotgun(self, num_iterations):
		m = self.evaluator.num_snps
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
		self.evaluator.get_probs()

	def reset_mapping(self):
		self.evaluator.reset()