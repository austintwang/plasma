from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np 
import scipy.linalg.lapack as lp
import itertools

class Evaluator(object):
	def __init__(
		self, causal_status_prior,
		imbalance_stats, imbalance_corr, imbalance_var_prior,
		total_exp_stats, total_exp_corr, total_exp_var_prior,
		cross_corr, cross_corr_prior
	):
		self.num_snps = np.shape(imbalance_stats)[0]
		self.causal_status_prior = causal_status_prior

		self.imbalance_stats = imbalance_stats
		self.total_exp_stats = total_exp_stats
		# self.stats = np.concatenate(imbalance_stats, total_exp_stats)

		self.imbalance_corr = imbalance_corr
		self.total_exp_corr = total_exp_corr
		self.cross_corr = cross_corr

		self.imbalance_var_prior = imbalance_var_prior
		self.total_exp_var_prior = total_exp_var_prior
		self.cross_cov_prior = cross_corr_prior * imbalance_var_prior * total_exp_var_prior

		# Pre-calculate values of $\Sigma_c^{-1}$
		imbalance_inv_var = 1.0 / imbalance_var_prior
		cross_cov = self.cross_cov_prior
		imbalance_schur = 1.0 / (total_exp_var_prior - cross_cov * imbalance_inv_var * cross_cov)
		self.inv_imbalance_prior = imbalance_inv_var + imbalance_schur * (imbalance_inv_var * cross_cov) ** 2 
		self.inv_cov_prior = -1 * imbalance_schur * imbalance_inv_var * cross_cov
		self.inv_total_exp_prior = imbalance_schur

		# Create structure for storing unnormalized results for each causal configuration
		self.results = {}
		self.cumu_sum = 0.0

		# Calculate result for null configuration
		null_config = np.zeros(self.num_snps)
		prior = (1 - 2 * self.causal_status_prior) ** self.num_snps
		self.results[tuple(null_config.tolist())] = float(prior)
		self.cumu_sum += float(prior)

	@staticmethod
	def _det(m):
		"""
		Returns the determinant of a symmetric positive-definite matrix 
		"""
		l = np.linalg.cholesky(m)
		prod_diag = l.diagonal().prod()
		return prod_diag ** 2

	@staticmethod
	def _inv(m):
		"""
		Returns the inverse of a symmetric positive-definite matrix
		"""
		l = np.linalg.cholesky(m)
		l_inv = lp.dtrtri(l)
		return l_inv.T * l_inv 

	def eval(self, configuration):
		key = tuple(configuration.tolist())

		if key in self.results:
			return self.results[key]

		num_causal = np.count_nonzero(configuration)
		prior_prob = (self.causal_status_prior ** num_causal * 
				(1 - 2 * self.causal_status_prior) ** (self.num_snps - num_causal))

		indices = configuration.nonzero()
		ind_2d = np.ix_(indices)

		stats = np.concatenate(self.imbalance_stats[indices], self.total_exp_stats[indices])
		cross_corr_ind = self.cross_corr[ind_2d]
		corr = np.concatenate(
			np.concatenate(self.imbalance_corr[ind_2d], cross_corr_ind, axis=0),
			np.concatenate(cross_corr_ind.T, self.total_exp_corr[ind_2d], axis=0)
			axis=1
		)

		prior_cov = np.concatenate(
			np.concatenate(
				np.eye(num_causal) * self.imbalance_var_prior, 
				np.eye(num_causal) * self.cross_cov_prior, 
				axis=0
			),
			np.concatenate(
				np.eye(num_causal) * self.cross_cov_prior, 
				np.eye(num_causal) * self.total_exp_var_prior, 
				axis=0
			)
			axis=1
		)
		prior_cov_inv = np.concatenate(
			np.concatenate(
				np.eye(num_causal) * self.inv_imbalance_prior, 
				np.eye(num_causal) * self.inv_cov_prior, 
				axis=0
			),
			np.concatenate(
				np.eye(num_causal) * self.inv_cov_prior, 
				np.eye(num_causal) * self.inv_total_exp_prior, 
				axis=0
			)
			axis=1
		)
		
		det = self._det(np.eye(2 * num_causal) + prior_cov * corr)
		inv = self._inv(prior_cov_inv + corr)
		bf = det ** -0.5 * np.exp(inv.dot(stats).dot(stats) / 2.0)
		res = prior_prob * bf

		self.results[key] = res
		self.cumu_sum += res
		return res

	def get_probs():
		probs = [(k, v / self.cumu_sum) for k, v in self.results.iteritems()]
		probs.sort(key=lambda x: x[1], reverse=True)
		return probs

	def clear_all():
		self.results = {}
		self.cumu_sum = 0.0


def search_exhaustive(evaluator, max_causal):
	m = evaluator.num_snps
	for k in xrange(max_causal):
		base = np.append(np.ones(k), np.zeros(m - k))
		for c in itertools.permutations(base):
			evaluator.eval(c)

def search_shotgun(evaluator, num_iterations):
	m = evaluator.num_snps
	configuration = np.zeros(m)
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
			posteriors.append(evaluator.eval(n))
		posteriors = np.array(posteriors)

		dist = posteriors / np.sum(posteriors)
		configuration = np.random.choice(neighbors, p=dist)







