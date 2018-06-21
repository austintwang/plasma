from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np 
import scipy.linalg.lapack as lp

class Evaluator(object):
	def __init__(self, fm):
		self.num_snps_imbalance = fm.num_snps_imbalance
		self.num_snps_total_exp = fm.num_snps_total_exp
		self.num_snps_combined = fm.num_snps_imbalance + fm.num_snps_total_exp
		self.num_snps = max(self.num_snps_imbalance, self.num_snps_total_exp)

		self.causal_status_prior = fm.causal_status_prior

		self.imbalance_stats = fm.imbalance_stats
		self.total_exp_stats = fm.total_exp_stats

		self.imbalance_corr = fm.imbalance_corr
		self.total_exp_corr = fm.total_exp_corr
		self.cross_corr = fm.cross_corr

		self.imbalance_var_prior = fm.imbalance_var_prior
		self.total_exp_var_prior = fm.total_exp_var_prior
		self.cross_cov_prior = fm.cross_corr_prior * fm.imbalance_var_prior * fm.total_exp_var_prior
		print(self.imbalance_var_prior) ####
		print(self.total_exp_var_prior) ####
		print(self.cross_cov_prior) ####
		
		# print(fm.imbalance_var_prior * (1 - fm.cross_corr_prior ** 2)) ####
		# print(fm.cross_corr_prior) ####
		self.imbalance_var_inv = (
			1 / (fm.imbalance_var_prior * (1 - fm.cross_corr_prior ** 2))
		)
		self.total_exp_var_inv = (
			1 / (fm.total_exp_var_prior * (1 - fm.cross_corr_prior ** 2))
		)
		self.cross_cov_inv = (
			-fm.cross_corr_prior 
			/ (
				np.sqrt(fm.imbalance_var_prior * fm.total_exp_var_prior) 
				* (1 - fm.cross_corr_prior ** 2)
			)
		)
		# print(self.imbalance_var_inv) ####
		# print(self.total_exp_var_inv) ####
		# print(self.cross_cov_inv) ####

		self.prior_cov = np.block([
			[
				np.eye(self.num_snps_imbalance) * self.imbalance_var_prior, 
				np.eye(self.num_snps_imbalance, self.num_snps_total_exp) * self.cross_cov_prior
			],
			[
				np.eye(self.num_snps_total_exp, self.num_snps_imbalance) * self.cross_cov_prior, 
				np.eye(self.num_snps_total_exp) * self.total_exp_var_prior,
			]
		])

		self.prior_cov_inv = np.block([
			[
				np.eye(self.num_snps_imbalance) * self.imbalance_var_inv, 
				np.eye(self.num_snps_imbalance, self.num_snps_total_exp) * self.cross_cov_inv
			],
			[
				np.eye(self.num_snps_total_exp, self.num_snps_imbalance) * self.cross_cov_inv, 
				np.eye(self.num_snps_total_exp) * self.total_exp_var_inv
			]
		])

		self.corr = np.block([
			[self.imbalance_corr, self.cross_corr.T],
			[self.cross_corr, self.total_exp_corr]
		])
		# print(self.corr) ####
		# print(self.cross_corr) ####
		# np.linalg.cholesky(self.corr) ####
		# print(all(np.diag(self.corr)==1)) ####
		# vals, vects = np.linalg.eig(self.corr) ####
		# print(list(vals)) ####
		# print(self.total_exp_corr) ####
		# np.linalg.cholesky(self.total_exp_corr) ####
		# print(self.total_exp_corr) ####


		self.det_term = np.eye(self.num_snps_combined) + np.matmul(self.prior_cov, self.corr)
		self.inv_term = self.prior_cov_inv + self.corr
		# print(self.det_term) ####

		self.stats = np.append(self.imbalance_stats, self.total_exp_stats)

		# print(self.imbalance_stats) ####
		# print(self.total_exp_stats) ####
		# print(np.isfinite(self.imbalance_stats)) ####

		self.valid_entries = np.append(
			np.isfinite(self.imbalance_stats), 
			np.isfinite(self.total_exp_stats)
		)

		# print(self.valid_entries) ####

		self.snp_map = np.append(
			np.arange(self.num_snps_imbalance),
			np.arange(self.num_snps_total_exp)
		)
		# print(self.snp_map) ####

		self.null_config = np.zeros(self.num_snps)

		# Create structure for storing unnormalized results for each causal configuration
		self.results = {}
		self.cumu_sum = 0.0

		# # Calculate result for null configuration
		# self.null_config = np.zeros(self.num_snps)
		# prior = (1 - 2 * self.causal_status_prior) ** self.num_snps
		# self.results[tuple(null_config.tolist())] = float(prior)
		# self.cumu_sum += float(prior)

	@staticmethod
	def _det(m):
		"""
		Returns the determinant of a symmetric positive-definite matrix 
		"""
		print(m) ####
		l = np.linalg.cholesky(m)
		prod_diag = l.diagonal().prod()
		return prod_diag ** 2

	@staticmethod
	def _inv(m):
		"""
		Returns the inverse of a symmetric positive-definite matrix
		"""
		l = np.linalg.cholesky(m)
		# print(l) ####
		l_inv = lp.dtrtri(l)
		# print(l_inv) ####
		return l_inv.T * l_inv 

	def eval(self, configuration, bias=1.0, prior=None):
		key = tuple(configuration.tolist())

		if key in self.results:
			return self.results[key] 

		if prior is None:
			num_causal = np.count_nonzero(configuration)
			prior = (
				self.causal_status_prior ** num_causal
				* (1 - 2 * self.causal_status_prior) ** (self.num_snps - num_causal)
			)

		if np.array_equal(configuration, self.null_config):
			res = prior / bias
			self.results[key] = res
			self.cumu_sum += res
			return res

		configuration_bool = (configuration == 1)
		selection = np.logical_and(configuration_bool[self.snp_map], self.valid_entries)
		if not np.any(selection):
			res = prior / bias
			self.results[key] = res
			self.cumu_sum += res
			return res

		selection_2d = np.ix_(selection, selection)

		# print(configuration) ####
		# print(configuration_bool) ####
		# print(list(selection)) ####

		det_term_subset = self.det_term[selection_2d]
		inv_term_subset = self.inv_term[selection_2d]
		stats_subset = self.stats[selection]

		# print(self.inv_term) ####
		# print(det_term_subset) ####
		# print(inv_term_subset) ####
		# print(stats_subset) ####

		det = self._det(det_term_subset)
		inv = self._inv(inv_term_subset)

		bf = (det ** -0.5) * np.exp(inv.dot(stats_subset).dot(stats_subset) / 2.0)
		res = bf * prior / bias

		self.results[key] = res
		self.cumu_sum += res
		return res

		# configuration_imbalance = configuration[:self.num_snps_imbalance]
		# configuration_total_exp = configuration[:self.num_snps_total_exp]
		# # print(configuration_imbalance) ####
		# # print(configuration_total_exp) ####

		# num_causal_imbalance = np.count_nonzero(configuration_imbalance)
		# num_causal_total_exp = np.count_nonzero(configuration_total_exp)
		# num_causal = np.count_nonzero(configuration)
		# prior_prob = (
		# 	self.causal_status_prior ** num_causal * 
		# 	(1 - 2 * self.causal_status_prior) ** (self.num_snps - num_causal)
		# )

		# indices_imbalance = configuration_imbalance.nonzero()[0]
		# indices_total_exp = configuration_total_exp.nonzero()[0]
		# # print(indices_imbalance) ####
		# # print(indices_total_exp) ####

		# ind_2d_imbalance = np.ix_(indices_imbalance, indices_imbalance)
		# ind_2d_total_exp = np.ix_(indices_total_exp, indices_total_exp)
		# ind_2d_cross = np.ix_(indices_total_exp, indices_imbalance)

		# # print(self.imbalance_stats[indices_imbalance]) ####
		# # print(self.total_exp_stats[indices_total_exp]) ####
		
		# stats = np.append(self.imbalance_stats[indices_imbalance], self.total_exp_stats[indices_total_exp])
		# cross_corr_ind = self.cross_corr[ind_2d_cross]
		# corr = np.append(
		# 	np.append(self.imbalance_corr[ind_2d_imbalance], cross_corr_ind.T, axis=1),
		# 	np.append(cross_corr_ind, self.total_exp_corr[ind_2d_total_exp], axis=1),
		# 	axis=0
		# )

		# prior_cov = np.append(
		# 	np.append(
		# 		np.eye(num_causal_imbalance) * self.imbalance_var_prior, 
		# 		np.eye(num_causal_imbalance, num_causal_total_exp) * self.cross_cov_prior, 
		# 		axis=1
		# 	),
		# 	np.append(
		# 		np.eye(num_causal_total_exp, num_causal_imbalance) * self.cross_cov_prior, 
		# 		np.eye(num_causal_total_exp) * self.total_exp_var_prior, 
		# 		axis=1
		# 	),
		# 	axis=0
		# )
		# prior_cov_inv = np.append(
		# 	np.append(
		# 		np.eye(num_causal_imbalance) * self.inv_imbalance_prior, 
		# 		np.eye(num_causal_imbalance, num_causal_total_exp) * self.inv_cov_prior, 
		# 		axis=1
		# 	),
		# 	np.append(
		# 		np.eye(num_causal_total_exp, num_causal_imbalance) * self.inv_cov_prior, 
		# 		np.eye(num_causal_total_exp) * self.inv_total_exp_prior, 
		# 		axis=1
		# 	),
		# 	axis=0
		# )
		
		# det = self._det(np.eye(num_causal_imbalance + num_causal_total_exp) + prior_cov * corr)
		# inv = self._inv(prior_cov_inv + corr)
		# bf = det ** -0.5 * np.exp(inv.dot(stats).dot(stats) / 2.0)
		# res = prior_prob * bf


	def get_probs(self):
		probs = copy(self.results)
		for k, v in probs.iteritems():
			v = v / self.cumu_sum
		return probs

	def get_probs_sorted(self):
		probs = [(k, v / self.cumu_sum) for k, v in self.results.viewitems()]
		probs.sort(key=lambda x: x[1], reverse=True)
		return probs

	def get_causal_set(self, confidence):
		configs = sorted(self.results, key=self.results.get, reverse=True)
		causal_set = [0] * max(self.num_snps_imbalance, self.num_snps_total_exp)
		threshold = confidence * self.cumu_sum
		conf_sum = 0
		for c in configs:
			causuality = self.results[configs]
			if conf_sum + causuality <= threshold:
				conf_sum += causuality
				for val, ind in enumerate(c):
					if val == 1:
						causal_set[ind] = 1
			else:
				break

		return causal_set

	def get_ppas(self):
		# m = max(self.num_snps_imbalance, self.num_snps_total_exp)
		ppas = []
		for i in xrange(self.num_snps):
			ppa = 0
			for k, v in self.results.iteritems():
				if k[i] == 1:
					ppa += v
			ppas.append(ppa / self.cumu_sum)
		return ppas

	def reset(self):
		self.results = {}
		self.cumu_sum = 0.0


