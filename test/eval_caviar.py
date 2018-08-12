from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import subprocess
import os
import shutil
import random
import string
import numpy as np

class EvalCaviar(object):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	caviar_path = os.path.join(dir_path, "caviar", "CAVIAR-C++", "CAVIAR")
	temp_path = os.path.join(dir_path, "temp")
	
	def __init__(self, fm, confidence, max_causal):
		self.num_snps = max(fm.num_snps_imbalance, fm.num_snps_total_exp)

		self.causal_status_prior = fm.causal_status_prior

		self.total_exp_stats = fm.total_exp_stats

		self.corr = fm.total_exp_corr

		self.var_prior = fm.imbalance_var_prior

		self.rsids = ["{0:05d}".format(i) for i in range(self.num_snps)]
		self.rsid_map = dict(zip(self.rsids, range(self.num_snps)))

		self.output_name = ''.join(
			random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
		)
		self.output_path = os.path.join(self.temp_path, self.output_name)
		os.mkdir(self.output_path)
		self.output_filename_base = os.path.join(self.output_path, self.output_name)

		self.z_path = os.path.join(self.output_path, "z.txt")
		self.ld_path = os.path.join(self.output_path, "ld.txt")
		self.set_path = os.path.join(self.output_path, self.output_name + "_set")

		self.params = [
			self.caviar_path,
			"-o", self.output_filename_base,
			"-l", self.ld_path,
			"-r", str(confidence),
			"-c", str(max_causal),
			"-n", str(self.var_prior)
		]

		self.causal_set = np.zeros(self.num_snps)

		self.z_scores = self.total_exp_stats.tolist()
		self.ld = self.corr.tolist()

	def run(self):
		with open(self.z_path, "w") as zfile:
			zstr = "\n".join("\t".join(str(j) for j in i) for i in zip(self.rsids, self.z_scores))
			zfile.write(zstr)

		with open(self.ld_path, "w") as ldfile:
			ldstr = "\n".join(" ".join(str(j) for j in i)for i in self.ld)
			ldfile.write(ldstr)

		subprocess.call(self.params)

		with open(self.set_path) as setfile:
			ids = setfile.read().splitlines()

		for i in ids:
			self.causal_set[self.rsid_map[int(i)]] = 1

		shutil.rmtree(self.output_path)

class EvalCaviarASE(object):
	def __init__(self, fm, confidence, max_causal):
		self.total_exp_stats = fm.total_exp_stats
		self.corr = fm.total_exp_corr

		self.imbalance = fm.imbalance
		self.phases = fm.phases

		self.phases_abs = np.absolute(self.phases)
		self.ase = np.logical_and(
			self.imbalance >= 0.619, self.imbalance <= -0.619
		)

		self.phases_imb = self.phases_abs.T[self.ase].T
		self.phases_bal = self.phases_abs.T[np.logical_not(self.ase)].T

		self.hets_tot = np.mean(self.phases_abs)
		self.hets_imb = np.mean(self.phases_imb)
		self.hets_bal = np.mean(self.phases_bal)

		self.num_imb = np.sum(self.ase) * 2
		self.num_bal = np.sum(1 - self.ase) * 2

		self.ase_std = np.sqrt(
			self.hets_tot 
			* (1 - self.hets_tot)
			* (self.num_bal + self.num_imb)
			/ (self.num_bal * self.num_imb)
		)

		self.ase_stats = (self.hets_imb - self.hets_bal) / self.ase_std

		self.stats_1 = (self.total_exp_stats + self.ase_stats) / np.sqrt(2)
		self.stats_2 = (self.total_exp_stats - self.ase_stats) / np.sqrt(2)

		means = np.mean(self.phases_abs, axis=0)
		phases_centered = self.phases_abs - means
		cov = phases_centered.T.dot(phases_centered)
		covdiag = np.diag(cov)
		denominator = np.sqrt(np.outer(covdiag, covdiag))
		corr = cov / denominator
		self.ld_ase = np.nan_to_num(corr)
		np.fill_diagonal(self.ld_ase, 1.0)

		self.ld = (self.corr + self.ld_ase) / 2

		self.eval1 = EvalCaviar(fm, confidence, max_causal)
		self.eval1.ld = self.ld.tolist()
		self.eval1.z_scores = self.stats_1.tolist()

		self.eval2 = EvalCaviar(fm, confidence, max_causal)
		self.eval2.ld = self.ld.tolist()
		self.eval2.z_scores = self.stats_2.tolist()

	def run(self):
		self.eval1.run()
		self.eval2.run()

		set1 = eval1.causal_set
		set2 = eval2.causal_set

		self.causal_set = min(set1, set2, key=np.size)



