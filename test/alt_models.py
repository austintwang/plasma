import subprocess
import os
import shutil
import random
import string
import numpy as np
import copy
import vcf

from . import Finemap

class Caviar(Finemap):
	cav_dir_path = "/agusevlab/awang/caviar"
	caviar_path = "CAVIAR"
	temp_path = os.path.join(cav_dir_path, "temp")
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def initialize(self):
		super().initialize()

		self.ncp = np.sqrt(self.imbalance_var_prior)

		self.rsids = ["rs{0:05d}".format(i) for i in range(self.num_snps)]
		self.rsid_map = dict(list(zip(self.rsids, list(range(self.num_snps)))))

		self.output_name = ''.join(
			random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
		)
		self.output_path = os.path.join(self.temp_path, self.output_name)
		os.makedirs(self.output_path)
		self.output_filename_base = os.path.join(self.output_path, self.output_name)

		self.z_path = os.path.join(self.output_path, "z.txt")
		self.ld_path = os.path.join(self.output_path, "ld.txt")
		self.set_path = os.path.join(self.output_path, self.output_name + "_set")
		self.post_path = os.path.join(self.output_path, self.output_name + "_post")

		self.causal_set = np.zeros(self.num_snps)
		self.post_probs = np.zeros(self.num_snps)

		self.z_scores = self.total_exp_stats.tolist()
		self.ld = self.corr.tolist()

	def search_exhuastive(self, min_causal, max_causal):
		self.min_causal = min_causal

	def search_shotgun(self, min_causal, max_causal, *args):
		self.search_exhuastive(min_causal, max_causal)

	def get_causal_set(self, confidence):
		self.params = [
			self.caviar_path,
			"-o", self.output_filename_base,
			"-l", self.ld_path,
			"-z", self.z_path,
			"-r", str(confidence),
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

		return self.causal_set

	def get_ppas(self):
		return self.post_probs


class CaviarASE(Finemap):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def initialize(self):
		super().initialize()

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

		self.ncp = 1

		self.eval1 = EvalCaviar(**kwargs)
		self.eval1.initialize()
		self.eval1.ld = self.ld.tolist()
		self.eval1.z_scores = self.stats_1.tolist()
		self.eval1.ncp = self.ncp

		self.eval2 = EvalCaviar(**kwargs)
		self.eval2.initialize()
		self.eval2.ld = self.ld.tolist()
		self.eval2.z_scores = self.stats_2.tolist()
		self.eval2.ncp = self.ncp

	def search_exhuastive(self, min_causal, max_causal):
		self.min_causal = min_causal
		self.eval1.search_exhuastive(min_causal, max_causal)
		self.eval2.search_exhuastive(min_causal, max_causal)

	def search_shotgun(self, min_causal, max_causal, *args):
		self.search_exhuastive(min_causal, max_causal)

	def get_causal_set(self, confidence):
		self.eval1.get_causal_set(confidence)
		self.eval2.get_causal_set(confidence)

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

	def get_ppas(self):
		return self.post_probs

class ECaviar(object):
	cav_dir_path = "/agusevlab/awang/caviar"
	caviar_path = "eCAVIAR"
	temp_path = os.path.join(cav_dir_path, "temp")
	
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
		self.rsid_map = dict(list(zip(self.rsids, list(range(self.num_snps)))))

		self.output_name = ''.join(
			random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
		)
		self.output_path = os.path.join(self.temp_path, self.output_name)
		os.makedirs(self.output_path)
		self.output_filename_base = os.path.join(self.output_path, self.output_name)
		# print(self.output_filename_base) ####

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
		self.post_probs_gwas = np.zeros(self.num_snps)
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

		with open(self.z_qtl_path, "w") as z_qtl_file:
			zstr = "\n".join("\t".join(str(j) for j in i) for i in zip(self.rsids, self.z_qtl)) + "\n"
			z_qtl_file.write(zstr)

		with open(self.z_gwas_path, "w") as z_gwas_file:
			zstr = "\n".join("\t".join(str(j) for j in i) for i in zip(self.rsids, self.z_gwas)) + "\n"
			z_gwas_file.write(zstr)

		with open(self.ld_qtl_path, "w") as ld_qtl_file:
			ldstr = "\n".join(" ".join(str(j) for j in i)for i in self.ld_qtl) + "\n"
			ld_qtl_file.write(ldstr)

		with open(self.ld_gwas_path, "w") as ld_gwas_file:
			ldstr = "\n".join(" ".join(str(j) for j in i)for i in self.ld_gwas) + "\n"
			ld_gwas_file.write(ldstr)

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

		# raise Exception ####
		shutil.rmtree(self.output_path)

class FmBenner(Finemap):
	fm_dir_path = "/agusevlab/awang/finemap"
	fm_path = "CAVIAR"
	temp_path = os.path.join(fm_dir_path, "temp")
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def initialize(self):
		super().initialize()

		self.ncp = np.sqrt(self.imbalance_var_prior)

		self.rsids = ["rs{0:05d}".format(i) for i in range(self.num_snps)]
		self.rsid_map = dict(list(zip(self.rsids, list(range(self.num_snps)))))

		self.output_name = ''.join(
			random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
		)
		self.output_path = os.path.join(self.temp_path, self.output_name)
		os.makedirs(self.output_path)
		self.output_filename_base = os.path.join(self.output_path, self.output_name)

		self.master_path = os.path.join(self.output_path, "master.txt")
		self.z_path = os.path.join(self.output_path, self.output_name + ".z")
		self.ld_path = os.path.join(self.output_path, self.output_name + ".ld")
		self.set_path = os.path.join(self.output_path, self.output_name + ".cred")
		self.post_path = os.path.join(self.output_path, self.output_name + ".snp")

		self.causal_set = np.zeros(self.num_snps)
		self.post_probs = np.zeros(self.num_snps)

		freq = (np.mean(self.hap_A, axis=0) + np.mean(self.hap_B, axis=0)) / 2.
		self.maf = np.fmin(freq, 1 - freq)
		self.betas = self.beta.tolist()
		self.se = (self.beta / self.imbalance_stats).tolist()
		self.ld = self.corr.tolist()

	def search_exhuastive(self, min_causal, max_causal):
		command_params = [
			self.fm_path,
			"--in-files", self.master_path,
			"--n-causal-snps", str(self.max_causal),
		]
		self._run_fm(command_params)

	def search_shotgun(self, min_causal, max_causal, prob_threshold, streak_threshold, num_iterations):
		command_params = [
			self.fm_path,
			"--in-files", self.master_path,
			"--n-causal-snps", str(self.max_causal),
			"--n-convergence", str(streak_threshold),
			"--n-iterations", str(num_iterations),
			"--prob-tol", str(prob_threshold),
			"--sss"
		]
		self._run_fm(command_params)

	def _run_fm(self, command_params):
		master_header = "z;ld;snp;config;cred;log;n_samples"
		master_content = "{0}.z;{0}.ld;{0}.snp;{0}.config;{0}.cred;{0}.log;{1}".format(self.output_name, self.num_ppl)
		with open(self.master_path, "w") as masterfile:
			masterfile.writelines([master_header, master_content])

		z_header = "rsid chromosome position allele1 allele2 maf beta se"
		z_template = "{0} 1 1 A T {1} {2} {3}"
		with open(self.z_path, "w") as zfile:
			zlines = [z_header]
			zlines.extend([z_template.format(*i) for i in zip(self.rsids, self.maf, self.betas, self.se)])
			zfile.writelines(zlines)

		with open(self.ld_path, "w") as ldfile:
			ldstr = "\n".join(" ".join(str(j) for j in i)for i in self.ld) + "\n"
			ldfile.write(ldstr)

		out = subprocess.check_output(command_params)

		with open(self.set_path) as setfile:
			ids = setfile.read().splitlines()[1].split()

		for i in ids:
			self.causal_set[self.rsid_map[i]] = 1

		post_df = pd.read_csv(self.post_path)
		self.post_probs = post_df["prob"].to_numpy()

		shutil.rmtree(self.output_path)

	def get_causal_set(self, confidence):
		return self.causal_set

	def get_ppas(self):
		return self.post_probs

class Rasqual(Finemap):
	rasqual_dir_path = "/agusevlab/awang/rasqual"
	rasqual_path = "rasqual"
	rasqual_script_path = "/agusevlab/awang/rasqual/R"
	r_path = "Rscript"
	temp_path = os.path.join(rasqual_dir_path, "temp")

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.vcf_reader = kwargs.get("vcf_reader", None)
		self.records = kwargs.get("records", None)
		self.snp_ids = kwargs.get("snp_ids", None)
		print("test1") ###

	def initialize(self):
		print("test2") ####
		self._calc_num_ppl()
		self._calc_num_snps()
		self._calc_causal_status_prior()
		self._calc_phases()
		self._calc_total_exp()
		self._calc_corr_stats()
		self._calc_imbalance_var_prior()
		self._calc_imbalance_corr()

		self.rsid_map = dict(list(zip(self.snp_ids, list(range(self.num_snps)))))

		self.output_name = ''.join(
			random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
		)
		self.output_path = os.path.join(self.temp_path, self.output_name)
		os.makedirs(self.output_path)
		self.output_filename_base = os.path.join(self.output_path, self.output_name)

		self.records_sim = copy.deepcopy(self.records)

		num_hets = np.count_nonzero(self.phases, axis=1)
		het_idx = np.argwhere(self.phases)

		for smp, snp in het_idx:
			record = self.records_sim[snp]
			sample = records_sim.samples[smp]
			gen_data = sample["GT"]
			hap_data = gen_data.split("|")

			reads = (0, 0)
			reads[hap_data[0]] = self.counts_A // num_hets[smp]
			reads[hap_data[1]] = self.counts_B // num_hets[smp]
			sample["AS"] = reads

		total_exp_scaled = self.total_exp * 50
		total_exp_off = total_exp_scaled - np.amain(total_exp_scaled)
		counts_data = "\t".join([self.output_name] + list(total_exp_off))

		self.vcf_path = os.path.join(self.output_path, "data.vcf")
		self.counts_path = os.path.join(self.output_path, "Y.txt")
		self.counts_bin_path = os.path.join(self.output_path, "Y.bin")
		self.offset_path = os.path.join(self.output_path, "K.txt")
		self.offset_bin_path = os.path.join(self.output_path, "K.bin")
		self.output_path = os.path.join(self.output_path, "output.txt")

		with open(self.vcf_path) as vcf_file:
			vcf_writer = vcf.Writer(vcf_file, self.vcf_reader)
			for record in self.records_sim:
				vcf_writer.write_record(record)

		with open(self.counts_path, "w") as counts_file:
			counts_file.writelines([counts_data])

		offset_script_path = os.path.join(rasqual_script_path, "makeOffset.R")
		offset_params = [
			self.r_path,
			"--vanilla",
			offset_script_path,
			self.offset_path,
			self.counts_path,
		]
		offset_out = subprocess.check_output(offset_params)

		bin_script_path = os.path.join(rasqual_script_path, "txt2bin.R")
		bin_params = [
			self.r_path,
			"--vanilla",
			bin_script_path,
			self.counts_path,
			self.offset_path,
		]
		bin_out = subprocess.check_output(bin_params)

		rasqual_params = [
			self.rasqual_path,
			self.vcf_path,
			"-y",
			self.counts_bin_path,
			"-k",
			self.offset_bin_path,
			"-n",
			str(self.num_ppl),
			"-j",
			"1",
			"-l",
			str(self.num_snps),
			"-m",
			str(self.num_snps),
			"-f",
			self.output_name,
		]
		print(" ".join(rasqual_params)) ####
		rasqual_out = subprocess.check_output(rasqual_params)
		rasqual_res = rasqual_out.decode('UTF-8').rstrip().split("\n")
		print("\n".join(rasqual_res)) ####

		z_scores = np.zeros(self.num_snps)
		for i in rasqual_res:
			entries = i.split("\t")
			rsid = entries[1]
			chisq = float(entries[10])
			pi = float(entries[11])

			if pi >= 0.5:
				z_scr = np.sqrt(chisq)
			else:
				z_scr = -np.sqrt(chisq)

			z_scores[self.rsid_map[rsid]] = z_scr

		self.imbalance_stats = z_scores

		shutil.rmtree(self.output_path)





	