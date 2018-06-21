from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np 
import scipy.linalg.lapack as lp
import itertools
import numbers

from .resources import FmUnchecked

class Finemap(FmUnchecked):
	def __init__(self, **kwargs):
		try:
			super(Finemap, self).__init__(**kwargs)
		except Exception:
			print("Error when instantiating Finemap object")
			raise

	@staticmethod
	def _check_number_type(number, name, min_type):
		if not isinstance(number, min_type):
			raise TypeError(
				"{0} is not an instance of {1}\nIs of type {2}:\n{3}".format(
					name, 
					str(min_type),
					str(type(number)),
					str(number)
				)
			)

	@staticmethod
	def _check_number_bounds(number, name, lower, upper):
		if number < lower or number > upper:
			print(number) ####
			raise ValueError(
				"{0} is not between {1} and {2}:\n{3}".format(
					name, 
					str(lower),
					str(upper),
					str(number),
				)
			)

	@staticmethod
	def _check_ndarray(matrix, name):
		if not isinstance(matrix, np.ndarray):
			raise TypeError(
				"{0} is not a Numpy array\nIs of type{1}:\n{2}".format(
					name,
					str(type(matrix)),
					str(matrix)
					)
			) 

	@staticmethod
	def _check_ndarray_dimensions(matrix, name, dimensions):
		if matrix.shape != dimensions:
			raise ValueError(
				"{0} is of incorrect dimensions\nExpected dimensions {1}\nGot dimensions {2}:\n{3}".format(
					name, 
					str(dimensions), 
					str(matrix.shape),
					str(matrix)
				)
			) 
	
	@staticmethod
	def _check_ndarray_dtype(matrix, name, min_type, empty):
		if empty:
			return
		if not issubclass(matrix.dtype.type, min_type):
			raise TypeError(
				"{0} does not consist of instances of {1}\nConsists of type {2}:\n{3}".format(
					name, 
					str(min_type),
					str(matrix.dtype),
					str(matrix)
				)
			)

	@staticmethod
	def _check_ndarray_bounds(matrix, name, lower, upper, empty):
		if empty:
			return
		if not ((matrix >= lower).all() and (matrix <= upper).all()):
			raise ValueError(
				"{0} contains element(s) not between {1} and {2}:\n{3}".format(
					name,
					str(lower),
					str(upper),
					str(matrix)
				)
			)

	def check_number(self, number, name):
		self._check_number_type(number, name, numbers.Real)

	def check_positive_number(self, number, name):
		self._check_number_type(number, name, numbers.Real)
		self._check_number_bounds(number, name, 0, float("inf"))

	def check_positive_int(self, number, name):
		self._check_number_type(number, name, numbers.Integral)
		self._check_number_bounds(number, name, 0, float("inf"))

	def check_probability(self, number, name):
		self._check_number_type(number, name, numbers.Real)
		self._check_number_bounds(number, name, 0, 1)

	def check_matrix(self, matrix, dimensions, name):
		empty = any(i == 0 for i in dimensions) 
		self._check_ndarray(matrix, name)
		self._check_ndarray_dimensions(matrix, name, dimensions)
		self._check_ndarray_dtype(matrix, name, numbers.Real, empty)

	def check_matrix_positive(self, matrix, dimensions, name):
		empty = any(i == 0 for i in dimensions) 
		self._check_ndarray(matrix, name)
		self._check_ndarray_dimensions(matrix, name, dimensions)
		self._check_ndarray_dtype(matrix, name, numbers.Real, empty)
		self._check_ndarray_bounds(matrix, name, 0, float("inf"), empty)

	def check_matrix_positive_int(self, matrix, dimensions, name):
		empty = any(i == 0 for i in dimensions) 
		self._check_ndarray(matrix, name)
		self._check_ndarray_dimensions(matrix, name, dimensions)
		self._check_ndarray_dtype(matrix, name, numbers.Integral, empty)
		self._check_ndarray_bounds(matrix, name, 0, float("inf"), empty)

	def check_matrix_corr(self, matrix, dimensions, name):
		empty = any(i == 0 for i in dimensions) 
		self._check_ndarray(matrix, name)
		self._check_ndarray_dimensions(matrix, name, dimensions)
		self._check_ndarray_dtype(matrix, name, numbers.Real, empty)

		if not np.allclose(matrix, matrix.T, atol=1e-8):
			raise ValueError(
				"{0} is not symmetric:\n{1}".format(
					name,
					str(matrix)
				)
			) 

		if not (np.diag(matrix) == 1).all():
			raise ValueError(
				"Diagonal values of {0} are not all 1:\n{1}".format(
					name,
					str(matrix)
				)
			)

	def check_matrix_01(self, matrix, dimensions, name):
		empty = any(i == 0 for i in dimensions) 
		self._check_ndarray(matrix, name)
		self._check_ndarray_dimensions(matrix, name, dimensions)
		self._check_ndarray_dtype(matrix, name, numbers.Integral, empty)
		self._check_ndarray_bounds(matrix, name, 0, 1, empty)

	def check_matrix_012(self, matrix, dimensions, name):
		empty = any(i == 0 for i in dimensions) 
		self._check_ndarray(matrix, name)
		self._check_ndarray_dimensions(matrix, name, dimensions)
		self._check_ndarray_dtype(matrix, name, numbers.Integral, empty)
		self._check_ndarray_bounds(matrix, name, 0, 2, empty)

	def check_matrix_n101(self, matrix, dimensions, name):
		empty = any(i == 0 for i in dimensions) 
		self._check_ndarray(matrix, name)
		self._check_ndarray_dimensions(matrix, name, dimensions)
		self._check_ndarray_dtype(matrix, name, numbers.Integral, empty)
		self._check_ndarray_bounds(matrix, name, -1, 1, empty)

	def _calc_counts(self):
		super(Finemap, self)._calc_counts()
		self.check_matrix_positive_int(
			self.counts_A, 
			(self.num_ppl_imbalance,), 
			"Haplotype A Read Counts"
		)
		self.check_matrix_positive_int(
			self.counts_B, 
			(self.num_ppl_imbalance,), 
			"Haplotype B Read Counts"
		)

	def _calc_haps(self):
		super(Finemap, self)._calc_haps()
		self.check_matrix_01(
			self.hap_A, 
			(self.num_ppl_total_exp, self.num_snps_total_exp,), 
			"Haplotype A Genotype Data"
		)
		self.check_matrix_positive_int(
			self.hap_B, 
			(self.num_ppl_total_exp, self.num_snps_total_exp,), 
			"Haplotype B Genotype Data"
		)

	def _calc_causal_status_prior(self):
		super(Finemap, self)._calc_causal_status_prior()
		self.check_probability(
			self.causal_status_prior, 
			"Prior probability of a causal configuration"
		)

	def _calc_imbalance(self):
		super(Finemap, self)._calc_imbalance()
		self.check_matrix(
			self.imbalance, 
			(self.num_ppl_imbalance,), 
			"Allelic imbalance phenotype vector"
		)

	def _calc_phases(self):
		super(Finemap, self)._calc_phases()
		self.check_matrix_n101(
			self.phases, 
			(self.num_ppl_imbalance, self.num_snps_imbalance,), 
			"Genotype phase matrix"
		)

	def _calc_total_exp(self):
		super(Finemap, self)._calc_total_exp()
		self.check_matrix(
			self.total_exp, 
			(self.num_ppl_total_exp,), 
			"Total expression phenotype vector"
		)

	def _calc_genotypes_comb(self):
		super(Finemap, self)._calc_genotypes_comb()
		self.check_matrix_012(
			self.genotypes_comb, 
			(self.num_ppl_total_exp, self.num_snps_total_exp,), 
			"Genotype allele count matrix"
		)

	def _calc_imbalance_errors(self):
		super(Finemap, self)._calc_imbalance_errors()
		self.check_matrix(
			self.imbalance_errors, 
			(self.num_ppl_imbalance,), 
			"Allelic imbalance error vector"
		)

	def _calc_imbalance_stats(self):
		super(Finemap, self)._calc_imbalance_stats()
		self.check_matrix(
			self.imbalance_stats, 
			(self.num_snps_imbalance,), 
			"Allelic imbalance association statistics vector"
		)

	def _calc_imbalance_corr(self):
		super(Finemap, self)._calc_imbalance_corr()
		self.check_matrix_corr(
			self.imbalance_corr, 
			(self.num_snps_imbalance, self.num_snps_imbalance), 
			"Allelic imbalance correlation matrix"
		)

	def _calc_beta(self):
		super(Finemap, self)._calc_beta()
		self.check_matrix(
			self._beta, 
			(self.num_snps_total_exp,), 
			"Total expression effect size vector"
		)
		self.check_number(self._mean, "Total expression mean")

	def _calc_total_exp_error(self):
		super(Finemap, self)._calc_total_exp_error()
		self.check_positive_number(self.exp_error_var, "Total expression error variance")

	def _calc_total_exp_stats(self):
		super(Finemap, self)._calc_total_exp_stats()
		self.check_matrix(
			self.total_exp_stats, 
			(self.num_snps_total_exp,), 
			"Total expression association statistics vector"
		)

	def _calc_total_exp_corr(self):
		super(Finemap, self)._calc_total_exp_corr()
		# print(self.total_exp_corr) ####
		self.check_matrix_corr(
			self.total_exp_corr, 
			(self.num_snps_total_exp, self.num_snps_total_exp), 
			"Total expression correlation matrix"
		)

	def _calc_cross_corr(self):
		super(Finemap, self)._calc_cross_corr()
		self.check_matrix(
			self.cross_corr, 
			(self.num_snps_total_exp, self.num_snps_imbalance), 
			"Cross-correlation matrix"
		)

	def initialize(self):
		self.check_positive_int(
			self.num_snps_imbalance, "Marker count for allelic imbalance data"
		)
		self.check_positive_int(
			self.num_snps_total_exp, "Marker count for total expression data"
		)
		self.check_positive_int(
			self.num_ppl_imbalance, "Number of individuals for allelic imbalance data"
		)
		self.check_positive_int(
			self.num_ppl_total_exp, "Number of individuals for total expression data"
		)
		# self.check_probability(
		# 	self.causal_status_prior, "Prior probability of a causal configuration"
		# )
		self.check_positive_number(
			self.imbalance_var_prior, "Prior variance for allelic imbalance"
		)
		self.check_positive_number(
			self.total_exp_var_prior, "Prior variance for total expression"
		)
		self.check_probability(
			self.cross_corr_prior, "Prior cross-correlation"
		)
		self.check_positive_number(
			self.overdispersion, "Allelic imbalance overdispersion"
		)

		super(Finemap, self).initialize()