from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np 
import scipy.stats
import vcf
import os
import random

class LocusSimulator(object):
	def __init__(
		self, 
		vcf_path, 
		chrom, 
		start, 
		num_snps, 
		sample_filter=None,
		maf_thresh=0.
	):
		vcf_reader = vcf.Reader(filename=vcf_path)
		samples = vcf_reader.samples
		if sample_filter is not None:
			filter_set = set(sample_filter)
			sample_idx = [ind for ind, val in enumerate(samples) if val in filter_set]
		else:
			sample_idx = range(len(samples))

		haps = []
		snp_ids = []
		snp_count = 0

		for record in vcf_reader.fetch(chrom, start, None):
			chr_num = record.CHROM
			pos = int(record.POS) + 1
			if record.ID == ".":
				snp_id = "{0}.{1}".format(chr_num, pos)
			else:
				snp_id = record.ID

			genotypes = []
			include_marker = True

			for ind in sample_idx:
				sample = record.samples[ind]

				gen_data = sample["GT"]
				if "/" in gen_data:
					include_marker = False
					break

				haps = gen_data.split("|")
				genotypes.append(int(haps[0]))
				genotypes.append(int(haps[1]))

			if include_marker:
				genotypes = np.array(genotypes)
				freq = np.mean(genotypes)
				maf = min(freq, 1 - freq)
				if maf < maf_thresh:
					include_marker = False

			if include_marker:
				haps.append(genotypes)
				snp_ids.append(snp_id)
				snp_count += 1

			if snp_count >= num_snps
				break

		self.haps = np.array(haps).T
		self.snp_ids = np.array(snp_ids)
		self.snp_count = snp_count

	def sim_asqtl(
			self, 
			num_samples,
			coverage,
			std_al_dev,
			herit_qtl,
			herit_as,
			overdispersion,
			num_causal
		):
		haps_idx = np.random.choice(np.shape(self.haps)[0], num_samples, replace=False)
		haps_sampled = self.haps[haps_idx]
		np.random.shuffle(haps_sampled)
		hap_A = haps_sampled[:num_samples]
		hap_B = haps_sampled[num_samples:]

		genotypes_comb = hap_A + hap_B
		phases = hap_A - hap_B

		causal_effects = npr.normal(0, 1, num_causal)
		causal_inds = npr.choice(self.num_snps, num_causal, replace=False)
		causal_config = np.zeros(self.num_snps)
		np.put(causal_config, causal_inds, 1)
		causal_snps = np.zeros(self.num_snps)
		np.put(causal_snps, causal_inds, causal_effects)

		prop_noise_eqtl = 1 - herit_qtl
		prop_noise_ase = 1 - herit_as

		exp_A = hap_A.dot(causal_snps)
		exp_B = hap_B.dot(causal_snps)

		imbalance_ideal = exp_A - exp_B
		imbalance_var = np.var(imbalance_ideal)
		imb_noise_var = imbalance_var * (prop_noise_ase / (1 - prop_noise_ase))
		imb_total_var = imbalance_var + imb_noise_var
		std_imbalance = np.log(std_al_dev) - np.log(1 - std_al_dev)
		imbalance = (
			npr.normal(imbalance_ideal, np.sqrt(imb_noise_var)) 
			* std_imbalance 
			/ np.sqrt(imb_total_var)
		)
		
		total_exp_ideal = exp_A + exp_B
		ideal_exp_var = np.var(total_exp_ideal)

		exp_noise_var = ideal_exp_var * (prop_noise_eqtl / (1 - prop_noise_eqtl))

		total_exp = npr.normal(total_exp_ideal, np.sqrt(exp_noise_var))
		
		betas = (1 / overdispersion - 1) * (1 / (1 + np.exp(imbalance)))
		alphas = (1 / overdispersion - 1) * (1 / (1 + np.exp(-imbalance)))

		@np.vectorize
		def _bb(counts, alpha, beta):
			p = npr.beta(alpha, beta, size=counts)
			return np.sum(npr.binomial(1, p))

		noised_coverage = npr.poisson(coverage, num_samples)
		noised_coverage[noised_coverage==0] = 1
		counts_A = _bb(noised_coverage, alphas, betas)

		counts_B = noised_coverage - counts_A
		counts_A[counts_A==0] = 1
		counts_B[counts_B==0] = 1

	def sim_gwas():

	dir_path = os.path.dirname(os.path.realpath(__file__))
	data_path = os.path.join(dir_path, "haplotypes")
	hap_name = "CEU.sampled_haplotypes"
	hap_path = os.path.join(data_path, hap_name)
	pickle_name = hap_name + ".pickle"
	pickle_path = os.path.join(data_path, pickle_name)
	num_haps = 190

	haps = None
	num_snps_total = None

	def __init__(self, params):
		if Haplotypes.haps is None:
			Haplotypes.load()

		self.num_snps = params["num_snps"]
		self.num_ppl = params["num_ppl"]
		if self.num_ppl > Haplotypes.num_haps // 2:
			raise ValueError("Not enough haplotypes to generate genotypes")
			
	@classmethod
	def load(cls):
		try:
			with open(cls.pickle_path, "rb") as hapfile:
				cls.haps = pickle.load(hapfile)
		except StandardError:
			# print("iosheiof") ####
			cls.build()
		finally:
			# print(cls.haps) ####
			cls.num_snps_total = cls.haps.shape[0]

	@classmethod
	def build(cls):
		# print("wheiieieiieiei") ####
		cls.hap_files = os.listdir(cls.hap_path)
		haps_list = [] 
		for f in cls.hap_files:
			# print("f") ####
			if f.endswith(".haps"):
				with open(os.path.join(cls.hap_path, f)) as hap:
					hapstr = hap.read()
				hap_block = [
					[int(j) for j in i.strip().split("\t")] 
					for i in hapstr.strip().split("\n")
				]
				for s in hap_block:
					if all(0 <= i <= 1 for i in s):
						prop_1 = sum(s) / len(s)
						if 0.01 <= prop_1 <=0.99:
							# print(len(s)) ####
							# print(prop_1) ####
							haps_list.append(s)
		cls.haps = np.array(haps_list)
		# print(cls.haps) ####
		with open(cls.pickle_path, "wb") as hapfile:
			pickle.dump(cls.haps, hapfile)
		# print("wheifhwoeihfowe") ####
		

		# self.haps = {}
		# self.hap_files = os.listdir(self.hap_path)
		# # print(self.hap_files) ####
		# for f in self.hap_files:
		# 	if f.endswith(".haps"):
		# 		with open(os.path.join(self.hap_path, f)) as hap:
		# 			hapstr = hap.read()
		# 		# print(hapstr) ####
		# 		# print([[int(j) for j in i.strip().split("\t")] for i in hapstr.strip().split("\n")]) ####
		# 		hap_arr = np.array(
		# 			[[int(j) for j in i.strip().split("\t")] for i in hapstr.strip().split("\n")]
		# 		).T
		# 		# print(hap_arr) ####
		# 		np.place(hap_arr, hap_arr>1, 0)
		# 		if hap_arr.shape[1] == self.NUM_SNPS_RAW:
		# 			self.haps[f] = hap_arr
		# with open(self.pickle_path, "wb") as hapfile:
		# 	pickle.dump(self.haps, hapfile)

	def draw_haps(self):
		start = np.random.randint(0, high=Haplotypes.num_snps_total-self.num_snps)
		end = start + self.num_snps
		section = np.arange(start, end)
		haps_section = Haplotypes.haps[section]

		locus_haps = haps_section.T

		# num_ppl = int(self.NUM_HAPS / 2)
		# locus = random.choice(list(self.haps))
		# locus_haps = self.haps[locus]
		a_ind = np.random.choice(Haplotypes.num_haps, self.num_ppl, replace=False)
		a_set = set(a_ind)
		b_ind = np.array([i for i in xrange(Haplotypes.num_haps) if i not in a_set])
		hapA = locus_haps[a_ind]
		hapB = locus_haps[b_ind]
		np.random.shuffle(hapA)
		np.random.shuffle(hapB)
		# print(a_ind) ####
		# print(b_ind) ####
		# # print(locus_haps) ####
		# haps_shuffled = np.random.shuffle(locus_haps)
		# # print(haps_shuffled) ####
		# hapA = haps_shuffled[:self.NUM_PPL]
		# hapB = haps_shuffled[self.NUM_PPL:]
		return hapA, hapB

def sim_locus
	vcf_reader = vcf.Reader(filename=chr_path)
	ppl_names = vcf_reader.samples
	num_ppl = len(ppl_names)