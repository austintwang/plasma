from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np 
import scipy.linalg.lapack as lp
import itertools
import numbers

from .resources import FM_unchecked

class Finemap(FM_unchecked):
	def __init__(self, **kwargs):
		try:
			super(Finemap, self).__init__(**kwargs)
		except Exception:
			print("Error when instantiating Finemap object")
			raise

	@staticmethod
	def check_number(number, name):
		if not isinstance(number, numbers.Real):
			raise TypeError(
				"{0} is not a real number\nIs of type {1}".format(
					name, str(type(number))
				)
			)

	@staticmethod
	def check_positive(number, name):
		if not isinstance(number, numbers.Real):
			raise TypeError(
				"{0} is not a real number\nIs of type {1}".format(
					name, str(type(number))
				)
			)
		if number < 0:
			raise TypeError(
				"{0} is between 0 and 1\nIs of value {1}".format(
					name, str(number)
				)
			)

	@staticmethod
	def check_positive_int(number, name):
		if not isinstance(number, numbers.Int):
			raise TypeError(
				"{0} is not an integer\nIs of type {1}".format(
					name, str(type(number))
				)
			)
		if number < 0:
			raise TypeError(
				"{0} is between 0 and 1\nIs of value {1}".format(
					name, str(number)
				)
			)

	
	@staticmethod
	def check_probability(number, name):
		if not isinstance(number, numbers.Real):
			raise TypeError(
				"{0} is not a real number\nIs of type {1}".format(
					name, str(type(number))
				)
			)
		if not 0 <= number <= 1:
			raise TypeError(
				"{0} is between 0 and 1\nIs of value {1}".format(
					name, str(number)
				)
			)

	@staticmethod
	def check_matrix(matrix, dimensions, name):
		if not isinstance(matrix, np.ndarray):
			raise TypeError("{0} is not a Numpy array".format(name)) 
		
		if not isinstance(matrix.dtype, numbers.Real):
			raise TypeError(
				"{0} does not consist of real numbers\nContains type {1}".format(
					name, str(matrix.dtype)
				)
			)

		if matrix.shape != dimensions:
			raise ValueError(
				"{0} is of incorrect dimensions\nExpected dimensions {1}\nGot dimensions {2}".format(
					name, dimensions, matrix.shape
				)
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
		self.check_matrix(
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
		self.check_matrix(
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
		self.check_matrix(
			self.imbalance_corr, 
			(self.num_snps_imbalance, self.num_snps_imbalance), 
			"Allelic imbalance correlation matrix"
		)

	def _calc_beta(self):
		super(Finemap, self)._calc_imbalance_stats()
		self.check_matrix(
			self._beta, 
			(self.num_ppl_total_exp,), 
			"Total expression effect size vector"
		)
		self.check_number(self._mean, "Total expression mean")

	def _calc_total_exp_error(self):
		super(Finemap, self)._calc_total_exp_errors()
		self.check_matrix(self.std_error, "Total expression standard error")

	def _calc_total_exp_stats(self):
		super(Finemap, self)._calc_total_exp_stats()
		self.check_matrix(
			self.total_exp_stats, 
			(self.num_snps_total_exp,), 
			"Total expression association statistics vector"
		)

	def _calc_total_exp_corr(self):
		super(Finemap, self)._calc_total_exp_corr()
		self.check_matrix(
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
		self.check_number(
			self.num_snps_imbalance, "Marker count for allelic imbalance data"
		)
		self.check_number(
			self.num_snps_total_exp, "Marker count for total expression data"
		)
		self.check_number(
			self.num_ppl_imbalance, "Number of individuals for allelic imbalance data"
		)
		self.check_number(
			self.num_ppl_total_exp, "Number of individuals for total expression data"
		)
		self.check_number(
			self.causal_status_prior, "Prior probability of a causal configuration"
		)
		self.check_number(
			self.imbalance_var_prior, "Prior probability of a causal configuration"
		)