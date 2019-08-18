import numpy as np 
from .reader import SubsetReader

class Universe(object):
	def __contains__(self, other):
		return True

class LocusSimulator(object):
	def __init__(
		self, 
		vcf_path, 
		chrom, 
		start, 
		num_causal, 
		region_size=None,
		max_snps=None,
		sample_filter=None,
		snp_filter=None,
		maf_thresh=0.
	):
		self.chrom = chrom
		self.start = start

		if (region_size is None) and (max_snps is None):
			raise ValueError("Must specifify either region_size or max_snps")

		if region_size is not None:
			self.region_size = region_size
			self.end = self.start + region_size
		else:
			self.region_size = np.inf
			self.end = None

		if max_snps is not None:
			self.max_snps = max_snps
		else:
			self.max_snps = np.inf

		if snp_filter is not None:
			# print("a") ####
			snp_filter = set(snp_filter)
			# print("b") ####
		else:
			snp_filter = Universe()

		if sample_filter is not None:
			sample_filter = set(sample_filter)
		else:
			sample_filter = Universe()	

		vcf_reader = SubsetReader(filename=vcf_path, snp_subset=snp_filter, sample_subset=sample_filter)

		records = []
		haps = []
		snp_ids = []
		snp_count = 0

		region = vcf_reader.fetch(self.chrom, self.start, self.end)
		for record in region:
			chr_rec = record.CHROM
			pos = int(record.POS) + 1

			if record.ID == ".":
				snp_id = "{0}.{1}".format(chr_rec, pos)
			else:
				snp_id = record.ID

			genotypes = []
			include_marker = True

			for sample in record.samples:
				gen_data = sample["GT"]
				if "/" in gen_data:
					include_marker = False
					break

				hap_data = gen_data.split("|")
				genotypes.append(int(hap_data[0]))
				genotypes.append(int(hap_data[1]))

			if not include_marker:
				continue

			genotypes = np.array(genotypes)
			freq = np.mean(genotypes)
			maf = min(freq, 1 - freq)
			if maf < maf_thresh:
				continue

			records.append(record)
			haps.append(genotypes)
			snp_ids.append(snp_id)
			snp_count += 1


			if snp_count >= self.max_snps:
				break

		if snp_count == 0:
			raise ValueError("Specified region yielded no markers")

		self.vcf_reader = vcf_reader
		self.records = records
		self.sim_end = pos

		self.haps = np.array(haps).T
		self.snp_ids = np.array(snp_ids)
		self.snp_count = snp_count
		self.num_samples = np.shape(self.haps)[0]

		causal_inds = np.random.choice(self.snp_count, num_causal, replace=False)
		self.causal_config = np.zeros(snp_count)
		np.put(self.causal_config, causal_inds, 1)
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
			causal_override=None,
			switch_error=0.,
			blip_error=0.,
		):
		mult = (num_samples * 2) // self.num_samples
		rem = (num_samples * 2) % self.num_samples
		blocks = []
		for _ in range(mult):
			blocks.append(np.arange(self.num_samples))
		blocks.append(np.random.choice(self.num_samples, rem, replace=False))
		haps_idx = np.concatenate(blocks)
		haps_sampled = self.haps[haps_idx]
		np.random.shuffle(haps_sampled)
		hap_A = haps_sampled[:num_samples]
		hap_B = haps_sampled[num_samples:]

		if switch_error > 0:
			switches = np.logical_and(
				(hap_A != hap_B), 
				np.random.choice([True, False], size=hap_A.shape(), p=[switch_error, 1-switch_error])
			)
			switch_idx = np.argwhere(switches)
			for r, c in switch_idx:
				hap_A[r, c:], hap_B[r, c:] = hap_B[r, c:], hap_A[r, c:].copy()

		if blip_error > 0:
			blips = np.logical_and(
				(hap_A != hap_B), 
				np.random.choice([True, False], size=hap_A.shape(), p=[blip_error, 1-blip_error])
			)
			blip_idx = np.argwhere(blips)
			for r, c in blip_idx:
				hap_A[r, c], hap_B[r, c] = hap_B[r, c], hap_A[r, c].copy()

		genotypes_comb = hap_A + hap_B
		phases = hap_A - hap_B
		
		if causal_override is not None:
			causal_config = causal_override
			num_causal = np.count_nonzero(causal_override)
		else:
			causal_config = self.causal_config
			num_causal = self.num_causal

		causal_effects = np.random.normal(0, 1, num_causal)
		causal_snps = np.zeros(self.snp_count)
		causal_snps[causal_config.astype(bool)] = causal_effects

		# print(herit_qtl) ####

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
			np.random.normal(imbalance_ideal, np.sqrt(imb_noise_var)) 
			* std_imbalance 
			/ np.sqrt(imb_total_var)
		)
		
		total_exp_ideal = exp_A + exp_B
		ideal_exp_var = np.var(total_exp_ideal)

		exp_noise_var = ideal_exp_var * (prop_noise_eqtl / (1 - prop_noise_eqtl))

		total_exp = np.random.normal(total_exp_ideal, np.sqrt(exp_noise_var))
		# print(total_exp_ideal) ####
		# print(total_exp) ####
		
		betas = (1 / overdispersion - 1) * (1 / (1 + np.exp(imbalance)))
		alphas = (1 / overdispersion - 1) * (1 / (1 + np.exp(-imbalance)))

		@np.vectorize
		def _bb(counts, alpha, beta):
			p = np.random.beta(alpha, beta, size=counts)
			return np.sum(np.random.binomial(1, p))

		noised_coverage = np.random.poisson(coverage, num_samples)
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
			causal_config = causal_override
			num_causal = np.count_nonzero(causal_override)
		else:
			causal_config = self.causal_config
			num_causal = self.num_causal

		gram = self.haps_cov * num_samples

		causal_effects = np.random.normal(0, 1, num_causal)
		causal_snps = np.zeros(self.snp_count)
		causal_snps[causal_config.astype(bool)] = causal_effects
		# print(causal_snps) ####

		var_causal_raw = causal_snps.dot(gram.dot(causal_snps)) / num_samples
		scale = herit / var_causal_raw
		causal_snps_scaled = causal_snps * np.sqrt(scale)

		# print(causal_snps_scaled) ####
		signal = gram.dot(causal_snps_scaled)
		# print(signal) ####
		noise = np.random.multivariate_normal(np.zeros(self.snp_count), gram * (1-herit))
		# print(noise) ####
		haps_var = np.diagonal(self.haps_cov)
		z_scores = (signal + noise) / np.sqrt(num_samples * haps_var * (1-herit))

		corr = self.haps_cov / np.sqrt(np.outer(haps_var, haps_var))
		corr = np.nan_to_num(corr)
		np.fill_diagonal(corr, 1.0)

		data_dict = {
			"z_gwas": z_scores,
			"ld_gwas": corr,
		}

		# raise Exception ####

		return data_dict
