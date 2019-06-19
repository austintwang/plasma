from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np 
import vcf

class LocusSimulator(object):
	def __init__(
		self, 
		vcf_path, 
		chrom, 
		start, 
		num_snps,
		num_causal, 
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

		self.causal_inds = np.random.choice(self.snp_count, num_causal, replace=False)
		self.num_causal = num_causal

		haps_means = np.mean(self.haps, axis=0)
		haps_centered = self.haps - haps_means
		self.haps_cov = np.nan_to_num(np.cov(haps_centered.T))

	def sim_asqtl(
			self, 
			num_samples,
			coverage,
			std_al_dev,
			herit_qtl,
			herit_as,
			overdispersion,
			causal_override=None
		):
		haps_idx = np.random.choice(np.shape(self.haps)[0], num_samples, replace=False)
		haps_sampled = self.haps[haps_idx]
		np.random.shuffle(haps_sampled)
		hap_A = haps_sampled[:num_samples]
		hap_B = haps_sampled[num_samples:]

		genotypes_comb = hap_A + hap_B
		phases = hap_A - hap_B
		
		if causal_override is not None:
			causal_inds = causal_override
			num_causal = np.size(causal_override)
		else:
			causal_inds = self.causal_inds
			num_causal = self.num_causal

		causal_effects = np.random.normal(0, 1, num_causal)
		causal_config = np.zeros(self.snp_count)
		np.put(causal_config, causal_inds, 1)
		causal_snps = np.zeros(self.snp_count)
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

		data_dict = {
			"total_exp": total_exp,
			"counts_A": counts_A,
			"counts_B": counts_B,
			"hap_A": hap_A,
			"hap_B": hap_B,
		}

		return data_dict

	def sim_gwas(
			self, 
			num_samples,
			herit,
			causal_override=None
		):
		if causal_override is not None:
			causal_inds = causal_override
			num_causal = np.size(causal_override)
		else:
			causal_inds = self.causal_inds
			num_causal = self.num_causal

		causal_effects = np.random.normal(0, 1, num_causal)
		var_causal_raw = causal_effects.dot(self.haps_cov.dot(causal_effects))
		scale = herit / var_causal_raw
		causal_effects_scaled = causal_effects * np.sqrt(scale)

		signal = self.haps_cov * causal_effects_scaled
		noise = np.random.multivariate_normal(0, self.haps_cov*(1-herit))
		haps_var = np.diagonal(self.haps_cov)
		z_scores = (signal + noise) * np.sqrt(self.snp_count / haps_var)

		return z_scores
