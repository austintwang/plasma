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
		self.confidence = confidence
		self.max_causal = max_causal

		self.num_snps = max(fm.num_snps_imbalance, fm.num_snps_total_exp)
		self.causal_status_prior = fm.causal_status_prior
		self.total_exp_stats = fm.total_exp_stats
		self.corr = fm.total_exp_corr
		self.ncp = np.sqrt(fm.imbalance_var_prior)

		self.rsids = ["rs{0:05d}".format(i) for i in range(self.num_snps)]
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
		self.post_path = os.path.join(self.output_path, self.output_name + "_post")

		self.causal_set = np.zeros(self.num_snps)
		self.post_probs = np.zeros(self.num_snps)

		self.z_scores = self.total_exp_stats.tolist()
		self.ld = self.corr.tolist()

	def run(self):
		self.params = [
			self.caviar_path,
			"-o", self.output_filename_base,
			"-l", self.ld_path,
			"-z", self.z_path,
			"-r", str(self.confidence),
			"-c", str(self.max_causal),
			"-n", str(self.ncp)
		]

		with open(self.z_path, "w") as zfile:
			zstr = "\n".join("\t".join(str(j) for j in i) for i in zip(self.rsids, self.z_scores)) + "\n"
			zfile.write(zstr)

		with open(self.ld_path, "w") as ldfile:
			ldstr = "\n".join(" ".join(str(j) for j in i)for i in self.ld) + "\n"
			ldfile.write(ldstr)

		out = subprocess.check_output(self.params)
		# print(out) ####
		# print(self.z_path) ####

		with open(self.set_path) as setfile:
			ids = setfile.read().splitlines()

		for i in ids:
			self.causal_set[self.rsid_map[i]] = 1

		with open(self.post_path) as postfile:
			posts = [i.split("\t") for i in postfile.read().splitlines()]
		postdict = {i[0]: i[2] for i in posts}

		for r in self.rsids:
			self.post_probs[self.rsid_map[r]] = postdict[r]

		shutil.rmtree(self.output_path)

class EvalCaviarASE(object):
	def __init__(self, fm, confidence, max_causal):
		self.total_exp_stats = fm.total_exp_stats
		self.corr = fm.total_exp_corr

		self.imbalance = fm.imbalance
		self.phases = fm.phases

		self.phases_abs = np.absolute(self.phases)
		self.ase = np.logical_or(
			self.imbalance >= 0.619, self.imbalance <= -0.619
		)

		self.phases_imb = self.phases_abs[self.ase]
		self.phases_bal = self.phases_abs[np.logical_not(self.ase)]

		self.hets_tot = np.mean(self.phases_abs, axis=0)
		self.hets_imb = np.mean(self.phases_imb, axis=0)
		self.hets_bal = np.mean(self.phases_bal, axis=0)

		self.num_imb = np.sum(self.ase) * 2
		self.num_bal = np.sum(1 - self.ase) * 2

		self.ase_std = np.sqrt(
			self.hets_tot 
			* (1 - self.hets_tot)
			* (self.num_bal + self.num_imb)
			/ (self.num_bal * self.num_imb)
		)
		# print(self.hets_tot) ####
		# print(self.num_bal) ####
		# print(self.num_imb) ####
		# print(self.ase_std) ####

		self.ase_stats = (self.hets_imb - self.hets_bal) / self.ase_std

		# print(self.ase_stats) ####
		# print(self.total_exp_stats) ####
		
		self.stats_1 = (self.total_exp_stats + self.ase_stats) / np.sqrt(2)
		self.stats_2 = (self.total_exp_stats - self.ase_stats) / np.sqrt(2)

		# print(self.stats_1) ####
		# print(self.stats_2) ####

		# print(fm.imbalance_stats) ####

		# raise Exception ####


		means = np.mean(self.phases_abs, axis=0)
		phases_centered = self.phases_abs - means
		cov = phases_centered.T.dot(phases_centered)
		covdiag = np.diag(cov)
		denominator = np.sqrt(np.outer(covdiag, covdiag))
		corr = cov / denominator
		self.ld_ase = np.nan_to_num(corr)
		np.fill_diagonal(self.ld_ase, 1.0)

		self.ld = (self.corr + self.ld_ase) / 2

		self.ncp = 1

		self.eval1 = EvalCaviar(fm, confidence, max_causal)
		self.eval1.ld = self.ld.tolist()
		self.eval1.z_scores = self.stats_1.tolist()
		self.eval1.ncp = self.ncp

		self.eval2 = EvalCaviar(fm, confidence, max_causal)
		self.eval2.ld = self.ld.tolist()
		self.eval2.z_scores = self.stats_2.tolist()
		self.eval2.ncp = self.ncp

	def run(self):
		self.eval1.run()
		self.eval2.run()

		set1 = self.eval1.causal_set
		set2 = self.eval2.causal_set

		ppa1 = self.eval1.post_probs
		ppa2 = self.eval2.post_probs

		if np.sum(set1) <= np.sum(set2):
			self.causal_set = set1
			self.post_probs = ppa1
		else:
			self.causal_set = set2
			self.post_probs = ppa2

class EvalECaviar(object):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	caviar_path = os.path.join(dir_path, "caviar", "CAVIAR-C++", "eCAVIAR")
	temp_path = os.path.join(dir_path, "temp")
	
	def __init__(self, fm_qtl, fm_gwas, confidence, max_causal):
		self.confidence = confidence
		self.max_causal = max_causal

		self.num_snps = fm_qtl.num_snps
		self.causal_status_prior = fm_qtl.causal_status_prior
		self.total_exp_stats_qtl = fm_qtl.total_exp_stats
		self.stats_gwas = fm_gwas.total_exp_stats
		self.corr_qtl = fm_qtl.total_exp_corr
		self.corr_gwas = fm_gwas.total_exp_corr
		# self.ncp = np.sqrt(fm.imbalance_var_prior)

		self.rsids = ["rs{0:05d}".format(i) for i in range(self.num_snps)]
		self.rsid_map = dict(zip(self.rsids, range(self.num_snps)))

		self.output_name = ''.join(
			random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
		)
		self.output_path = os.path.join(self.temp_path, self.output_name)
		os.mkdir(self.output_path)
		self.output_filename_base = os.path.join(self.output_path, self.output_name)

		self.z_qtl_path = os.path.join(self.output_path, "z_qtl.txt")
		self.z_gwas_path = os.path.join(self.output_path, "z_gwas.txt")
		self.ld_qtl_path = os.path.join(self.output_path, "ld_qtl.txt")
		self.ld_gwas_path = os.path.join(self.output_path, "ld_gwas.txt")
		self.set_qtl_path = os.path.join(self.output_path, self.output_name + "_1_set")
		self.set_gwas_path = os.path.join(self.output_path, self.output_name + "_2_set")
		self.post_qtl_path = os.path.join(self.output_path, self.output_name + "_1_post")
		self.post_gwas_path = os.path.join(self.output_path, self.output_name + "_2_post")
		self.clpp_path = os.path.join(self.output_path, self.output_name + "_col")

		self.causal_set_qtl = np.zeros(self.num_snps)
		self.causal_set_gwas = np.zeros(self.num_snps)
		self.post_probs_qtl = np.zeros(self.num_snps)
		self.causal_set_gwas = np.zeros(self.num_snps)
		self.clpp = np.zeros(self.num_snps)

		self.z_qtl = self.total_exp_stats_qtl.tolist()
		self.z_gwas = self.stats_gwas.tolist()
		self.ld_qtl = self.corr_qtl.tolist()
		self.ld_gwas = self.corr_gwas.tolist()

	def run(self):
		self.params = [
			self.caviar_path,
			"-o", self.output_filename_base,
			"-l", self.ld_qtl_path,
			"-l", self.ld_gwas_path,
			"-z", self.z_qtl_path,
			"-z", self.z_gwas_path,
			"-r", str(self.confidence),
			"-c", str(self.max_causal),
			# "-n", str(self.ncp)
		]

		with open(self.z_path, "w") as zfile:
			zstr = "\n".join("\t".join(str(j) for j in i) for i in zip(self.rsids, self.z_scores)) + "\n"
			zfile.write(zstr)

		with open(self.ld_path, "w") as ldfile:
			ldstr = "\n".join(" ".join(str(j) for j in i)for i in self.ld) + "\n"
			ldfile.write(ldstr)

		out = subprocess.check_output(self.params)
		# print(out) ####
		# print(self.z_path) ####

		with open(self.set_qtl_path) as setfile_qtl:
			ids_qtl = setfile_qtl.read().splitlines()

		for i in ids_qtl:
			self.causal_set_qtl[self.rsid_map[i]] = 1

		with open(self.set_gwas_path) as setfile_gwas:
			ids_gwas = setfile_gwas.read().splitlines()

		for i in ids_gwas:
			self.causal_set_gwas[self.rsid_map[i]] = 1

		with open(self.post_qtl_path) as postfile_qtl:
			posts_qtl = [i.split("\t") for i in postfile_qtl.read().splitlines()]
		postdict_qtl = {i[0]: i[2] for i in posts_qtl}

		for r in self.rsids:
			self.post_probs_qtl[self.rsid_map[r]] = postdict_qtl[r]

		with open(self.post_gwas_path) as postfile_gwas:
			posts_gwas = [i.split("\t") for i in postfile_gwas.read().splitlines()]
		postdict_gwas = {i[0]: i[2] for i in posts_gwas}

		for r in self.rsids:
			self.post_probs_gwas[self.rsid_map[r]] = postdict_gwas[r]

		with open(self.clpp_path) as clppfile:
			clpps = [i.split("\t") for i in clppfile.read().splitlines()]
		clppdict_gwas = {i[0]: i[2] for i in clpps}

		for r in self.rsids:
			self.clpp[self.rsid_map[r]] = clppdict_gwas[r]

		self.h4 = np.sum(self.clpp)

		shutil.rmtree(self.output_path)

