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
		self.cross_corr_prior = fm.cross_corr_prior
		self.cross_cov_prior = (
			self.cross_corr_prior 
			* np.sqrt(fm.imbalance_var_prior * fm.total_exp_var_prior)
		)
		# print(self.imbalance_var_prior) ####
		# print(self.total_exp_var_prior) ####
		# print(self.cross_cov_prior) ####
		
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

		# self.prior_cov = np.block([
		# 	[
		# 		np.eye(self.num_snps_imbalance) * self.imbalance_var_prior, 
		# 		np.eye(self.num_snps_imbalance, self.num_snps_total_exp) * self.cross_cov_prior
		# 	],
		# 	[
		# 		np.eye(self.num_snps_total_exp, self.num_snps_imbalance) * self.cross_cov_prior, 
		# 		np.eye(self.num_snps_total_exp) * self.total_exp_var_prior,
		# 	]
		# ])

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
		# print("eigs") ####
		# print(vals.real) ####
		# vals, vects = np.linalg.eig(self.imbalance_corr) ####
		# print("eigs imb") ####
		# print(vals.real) ####
		# vals, vects = np.linalg.eig(self.total_exp_corr) ####
		# print("eigs tot") ####
		# print(vals.real) ####
		# vals, vects = np.linalg.eig(self.cross_corr) ####
		# print("eigs cross") ####
		# print(vals.real) ####
		# vals, vects = np.linalg.eig(self.prior_cov_inv) ####
		# print("eigs2") ####
		# print(vals.real) ####
		# print(self.total_exp_corr) ####
		# np.linalg.cholesky(self.total_exp_corr) ####
		# np.linalg.cholesky(self.imbalance_corr) ####
		# print(self.total_exp_corr) ####
		# np.linalg.cholesky(self.corr) ####
		# np.linalg.cholesky(self.prior_cov) ####
		# np.linalg.cholesky(self.prior_cov_inv) ####


		# self.det_term = np.eye(self.num_snps_combined) + np.matmul(self.prior_cov, self.corr)
		self.inv_term = self.prior_cov_inv + self.corr
		# print(self.det_term) ####
		# self.chol = np.linalg.cholesky(self.inv_term) ####
		# vals, vects = np.linalg.eig(self.prior_cov_inv * 1 + self.corr) ####
		# print("eigs3") ####
		# print(vals.real) ####
		# print(np.nonzero(vals.real <= 0)[0]) ####

		# raise Exception ####

		self.stats = np.append(self.imbalance_stats, self.total_exp_stats)

		# print(self.imbalance_stats) ####
		# print(self.total_exp_stats) ####
		# print(np.isfinite(self.imbalance_stats)) ####

		if self.num_snps_imbalance == 0:
			self.ldet_prior = np.log(self.total_exp_var_prior)
		elif self.num_snps_total_exp == 0:
			self.ldet_prior = np.log(self.imbalance_var_prior)
		else:
			self.ldet_prior = (
				np.log(self.imbalance_var_prior)
				+ np.log(self.total_exp_var_prior)
				+ np.log(1 - self.cross_corr_prior ** 2)
			)
		# print(self.ldet_prior) ####
		# print(self.cross_cov_prior) ####

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
		self.results_unsaved = {}
		self.cumu_lposts = None
		# self.cumu_sum = 0.0

		# # Calculate result for null configuration
		# self.null_config = np.zeros(self.num_snps)
		# prior = (1 - 2 * self.causal_status_prior) ** self.num_snps
		# self.results[tuple(null_config.tolist())] = float(prior)
		# self.cumu_sum += float(prior)


	# @staticmethod
	# def _det(m):
	# 	"""
	# 	Returns the determinant of a symmetric positive-definite matrix 
	# 	"""
	# 	# print(m) ####
	# 	# vals, vects = np.linalg.eig(m) ####
	# 	# print("eigs") ####
	# 	# print(list(vals)) ####
	# 	l = np.linalg.cholesky(m)
	# 	prod_diag = l.diagonal().prod()
	# 	return prod_diag ** 2

	# @staticmethod
	# def _inv(m):
	# 	"""
	# 	Returns the inverse of a symmetric positive-definite matrix
	# 	"""
	# 	# print(m) ####
	# 	l = np.linalg.cholesky(m)
	# 	# print(l) ####
	# 	# print(np.linalg.inv(l)) ####
	# 	l_inv = lp.dtrtri(l, lower=1)[0]
	# 	# print(l_inv) ####
	# 	return np.matmul(l_inv.T, l_inv)

	@staticmethod
	def _lbf(cov_term, stats, ldet_prior):
		ltri = np.linalg.cholesky(cov_term)
		ldet_term = np.log(np.prod(np.diagonal(ltri))) * 2
		temp = np.asfortranarray(stats[:, np.newaxis])
		lp.dtrtrs(ltri, temp, lower=1, overwrite_b=1)
		# print(stats.size) ####
		# print(temp ** 2) ####
		return (-ldet_term - ldet_prior * stats.size + np.sum(temp ** 2)) / 2

	def eval(self, configuration, lbias=0.0, lprior=None, save_result=True):
		key = tuple(configuration.tolist())

		if key in self.results:
			return self.results[key] 
		elif key in self.results_unsaved:
			return self.results_unsaved[key] 

		if lprior is None:
			num_causal = np.count_nonzero(configuration)
			lprior = (
				np.log(self.causal_status_prior) * num_causal
				+ np.log(1 - 2 * self.causal_status_prior) * (self.num_snps - num_causal)
			)

		if np.array_equal(configuration, self.null_config):
			res = lprior - lbias
			self.results[key] = res
			# self.cumu_sum += res
			return res

		configuration_bool = (configuration == 1)
		selection = np.logical_and(configuration_bool[self.snp_map], self.valid_entries)
		if not np.any(selection):
			res = lprior - lbias
			self.results[key] = res
			# self.cumu_sum += res
			return res

		selection_2d = np.ix_(selection, selection)

		# print(configuration) ####
		# print(configuration_bool) ####
		# print(list(selection)) ####

		# det_term_subset = self.det_term[selection_2d]
		inv_term_subset = self.inv_term[selection_2d]
		stats_subset = self.stats[selection]

		# print(self.prior_cov_inv[selection_2d]) ####
		# print(np.linalg.inv(self.prior_cov[selection_2d])) ####
		# if not np.allclose(self.prior_cov_inv[selection_2d] + 0.0, np.linalg.inv(self.prior_cov[selection_2d])): ####
		# 	print(self.prior_cov_inv[selection_2d]) ####
		# 	print(np.linalg.inv(self.prior_cov[selection_2d])) ####
		# 	print(self.prior_cov[selection_2d]) ####
		# 	print(self.imbalance_var_prior) ####
		# 	print(self.total_exp_var_prior) ####
		# 	print(self.cross_cov_prior) ####
		# 	print(self.imbalance_var_inv) ####
		# 	print(self.total_exp_var_inv) ####
		# 	print(self.cross_cov_inv) ####
		# 	raise Exception

		# subchol = np.linalg.cholesky(inv_term_subset) ####
		# assert(subchol == self.chol[selection_2d]) ####
		# print(subchol) ####
		# print(self.chol[selection_2d]) ####
		# print(configuration) ####
		# print("") ####

		# print(self.inv_term) ####
		# print(det_term_subset) ####
		# print(inv_term_subset) ####
		# print(stats_subset) ####
		# det = np.linalg.det(det_term_subset)
		# inv = lp.dpotri(inv_term_subset)[0]
		# inv = self._inv(inv_term_subset)
		# try: 
		# 	inv = self._inv(inv_term_subset)
		# 	# det = self._det(det_term_subset)
		# except np.linalg.linalg.LinAlgError as err: 
		# 	# print(det_term_subset) ####
		# 	print(inv_term_subset) ####
		# 	print(stats_subset) ####
		# 	vals, vects = np.linalg.eig(inv_term_subset) ####
		# 	print(vals)
		# 	# print(selection) ####
		# 	raise
		# print(stats_subset) ####
		# print(det) ####
		# print(np.linalg.det(det_term_subset)) ####
		# print(inv_term_subset) ####
		# print(inv) ####
		# print(np.linalg.inv(inv_term_subset)) ####
		# print(det ** -0.5) ####
		# print(inv.dot(stats_subset)) ####
		# print(inv.dot(stats_subset).dot(stats_subset)) ####
		# print(np.exp(inv.dot(stats_subset).dot(stats_subset) / 2.0)) ####

		# dist = inv.dot(stats_subset).dot(stats_subset)
		# ldet = np.linalg.slogdet(det_term_subset)[1]
		# lbf = ldet * -0.5 + (dist / 2.0)
		# corr = np.log(1 + dist**4 + dist**3 / 4)
		# print(corr) ####

		# lmvn = self._eval_lmvn(inv_term_subset, stats_subset)
		lbf = self._lbf(inv_term_subset, stats_subset, self.ldet_prior)

		res = lbf + lprior - lbias

		# if res == np.nan: ####
		# 	raise Exception ####

		# print(bf) ####
		# print(res) ####
		# print("") ####

		if save_result:
			self.results[key] = res
			if self.cumu_lposts is None:
				self.cumu_lposts = res
			else:
				self.cumu_lposts += np.log(1 + np.exp(res - self.cumu_lposts))
		else:
			self.results_unsaved[key] = res
		# self.cumu_sum += res
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
		probs = {}
		total_lpost = self.cumu_lposts
		# max_lbf = max(self.results.values())
		# scale = 25 - max_lbf
		# total = sum([np.exp(i + scale) for i in self.results.values()])
		for k, v in self.results.viewitems():
			lprob = v - total_lpost
			if lprob > 0: ####
				print(v, total_lpost, np.exp(lprob)) ####
				raise Exception ####
			probs[k] = np.exp(lprob)
			# print(v) ####
			# print(probs[k]) ####
		return probs

	def get_probs_sorted(self):
		probs = self.get_probs().items()
		probs.sort(key=lambda x: x[1], reverse=True)
		return probs

	def get_causal_set(self, confidence):
		# max_lbf = max(self.results.values())
		# scale = 25 - max_lbf
		# total = sum([np.exp(i + scale) for i in self.results.values()])
		# threshold = confidence * total
		# print(self.get_ppas()) ####
		# configs = sorted(self.results, key=self.results.get, reverse=True)
		# causal_set = [0] * self.num_snps
		# conf_sum = 0
		# for c in configs:
		# 	causuality = np.exp(self.results[c] + scale)
		# 	# print(causuality) ####
		# 	conf_sum += causuality
		# 	# print(conf_sum, causuality) ####
		# 	# print(conf_sum / total) ####
		# 	# print(c) ####
		# 	for ind, val in enumerate(c):
		# 		if val == 1:
		# 			causal_set[ind] = 1
		# 	if conf_sum >= threshold:
		# 		# print(conf_sum / threshold) ####
		# 		break
		# confidence = 0 ####
		# confidence = np.inf ####
		
		# results_exp = {k: np.exp(v + scale) for k, v in self.results.viewitems()}
		# total = sum(results_exp.values())
		# threshold = confidence * total

		results_exp = self.get_probs()

		causal_set = np.zeros(self.num_snps)
		# causal_set = tuple([0] * self.num_snps)
		conf_sum = results_exp.get(tuple(causal_set), 0.)

		distances = {}
		causal_extras = {}
		for k in self.results.keys():
			causals = set(ind for ind, val in enumerate(k) if val == 1)
			distances.setdefault(sum(k), set()).add(k)
			causal_extras[k] = causals
		# print(distances) ####
		print(conf_sum) ####

		while conf_sum < confidence:
			dist_ones = distances[1]
			neighbors = {}
			for i in dist_ones:
				# results_exp[i[0]] ####
				# print(i) ####
				# print(causal_extras[i]) ####
				diff_snp = next(iter(causal_extras[i]))
				neighbors.setdefault(diff_snp, 0)
				neighbors[diff_snp] += results_exp[i]

			max_snp = max(neighbors, key=neighbors.get)
			causal_set[max_snp] = 1
			conf_sum += neighbors[max_snp]
			print(conf_sum) ####

			diffs = {}
			for k, v in distances.viewitems():
				diffs[k] = set() 
				for i in v:
					if i[max_snp] == 1:
						diffs[k].add(i)
						if k == 1:
							causal_extras.pop(i)
						else:
							causal_extras[i].remove(max_snp)

			for k, v in diffs.viewitems():
				distances[k] -= v
				if k > 1:
					distances.setdefault(k-1, set())
					distances[k-1] |= v

		# print(causal_set) ####
		return list(causal_set) 

		# causal_set = [0] * self.num_snps
		# conf_sum = 0
		# sets_considered = {i: [] for i in xrange(self.num_snps)}
		# sums_considered = {i: 0 for i in xrange(self.num_snps)}
		# while conf_sum < threshold:
		# 	for k, v in results_exp.viewitems():
		# 		diffs = False
		# 		addition = None
		# 		ignore = False
		# 		for ind, val in enumerate(k):
		# 			if val > causal_set[ind]:
		# 				if diffs:
		# 					ignore = True
		# 					break
		# 				diffs = True
		# 				addition = ind 
		# 		if ignore:
		# 			continue
		# 		sets_considered[addition].append(k)
		# 		sums_considered[addition] += v

		# 	max_snp = max(sums_considered, key=sums_considered.get)
		# 	causal_set[max_snp] = 1
		# 	sets_used = sets_considered.pop(max_snp)
		# 	# print(sets_used) ####
		# 	conf_sum += sums_considered.pop(max_snp)

		# 	for s in sets_used:
		# 		results_exp.pop(s, None)
		# 	for k in sets_considered.keys():
		# 		sets_considered[k] = []
		# 		sums_considered[k] = 0

		# 	# print(conf_sum / total) ####
		# 	# print(len(results_exp)) ####
		# 	# print(sets_considered) ####
		# 	# print(sums_considered) ####


		# causal_set = [1] * self.num_snps
		# conf_sum = total
		
		# sum_remaining = {k: 0 for k in xrange(self.num_snps)}
		# configs_remaining = {k: set() for k in xrange(self.num_snps)}
		# for k, v in results_exp.viewitems():
		# 	for ind, val in enumerate(k):
		# 		if val == 1:
		# 			configs_remaining[ind].add(k)
		# 			sum_remaining[ind] += v

		# while conf_sum >= threshold:
		# 	min_snp = min(sum_remaining, key=sum_remaining.get)
		# 	sum_new = conf_sum - sum_remaining.pop(min_snp)
		# 	if sum_new < threshold:
		# 		break

		# 	conf_sum = sum_new
		# 	configs_min = configs_remaining.pop(min_snp)
		# 	causal_set[min_snp] = 0

		# 	for ind, val in configs_remaining.viewitems():
		# 		configs_used = configs_min & val
		# 		sum_used = 0
		# 		for c in configs_used:
		# 			sum_used += results_exp[c]
				
		# 		sum_remaining[ind] -= sum_used
		# 		val -= configs_used

			# print(conf_sum / total) ####

		# print("------") ####
		# raise Exception ####
		return causal_set

	def get_ppas(self):
		# m = max(self.num_snps_imbalance, self.num_snps_total_exp)
		# total = sum(i for i in self.get_probs().values())
		# print(total) ####
		ppas = []
		for i in xrange(self.num_snps):
			ppa = 0
			for k, v in self.get_probs().viewitems():
				if k[i] == 1:
					ppa += v
			ppas.append(ppa)
		return ppas

	def get_size_probs(self):
		size_probs = np.zeros(self.num_snps)
		for k, v in self.get_probs().viewitems():
			num_snps = np.count_nonzero(k)
			size_probs[num_snps] += v
		return size_probs

	def reset(self):
		self.results = {}
		# self.cumu_sum = 0.0


