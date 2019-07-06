import numpy as np 
import vcf
from vcf.model import _Record

class SubsetReader(vcf.Reader):
	def __init__(self, **kwargs):
		self.snp_subset = kwargs.pop("snp_subset")
		self.sample_subset = kwargs.pop("sample_subset")
		super().__init__(**kwargs)

	def _parse_metainfo(self):
		super()._parse_metainfo()
		self.sample_pos = [ind for ind, val in enumerate(self.samples) if val in self.sample_subset]
		self.samples = [i for i in self.samples if i in self.sample_subset]
		self._sample_indexes = {k: v for k, v in self._sample_indexes.items() if k in self.sample_subset}

	def _parse_samples(self, samples, samp_fmt, site):
		samples_sel = [samples[i] for i in self.sample_pos]
		return super()._parse_samples(samples_sel, samp_fmt, site)

	def _parse_filter(self, filt_str):
		if filt_str == '.':
			return None
		elif filt_str == 'PASS':
			return []
		else:
			return filt_str.split(';')

	def __next__(self):
		'''Return the next record in the file.'''
		while True:
			line = next(self.reader)
			row = self._row_pattern.split(line.rstrip())

			if row[2] != '.':
				ID = row[2]
			else:
				ID = None

			if ID in self.snp_subset:
				break

		chrom = row[0]
		if self._prepend_chr:
			chrom = 'chr' + chrom
		pos = int(row[1])

		ref = row[3]
		alt = self._map(self._parse_alt, row[4].split(','))

		try:
			qual = int(row[5])
		except ValueError:
			try:
				qual = float(row[5])
			except ValueError:
				qual = None

		filt = self._parse_filter(row[6])
		info = self._parse_info(row[7])

		try:
			fmt = row[8]
		except IndexError:
			fmt = None
		else:
			if fmt == '.':
				fmt = None

		record = _Record(chrom, pos, ID, ref, alt, qual, filt,
				info, fmt, self._sample_indexes)

		if fmt is not None:
			samples = self._parse_samples(row[9:], fmt, record)
			record.samples = samples

		return record


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
			snp_filter = set(snp_filter)
		else:
			snp_filter = Universe()

		if sample_filter is not None:
			sample_filter = set(sample_filter)
		else:
			sample_filter = Universe()	

		vcf_reader = SubsetReader(filename=vcf_path, snp_subset=snp_filter, sample_subset=sample_filter)

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

			haps.append(genotypes)
			snp_ids.append(snp_id)
			snp_count += 1


			if snp_count >= self.max_snps:
				break

		if snp_count == 0:
			raise ValueError("Specified region yielded no markers")

		self.haps = np.array(haps).T
		self.snp_ids = np.array(snp_ids)
		self.snp_count = snp_count

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
			causal_override=None
		):
		haps_idx = np.random.choice(np.shape(self.haps)[0], num_samples * 2, replace=False)
		haps_sampled = self.haps[haps_idx]
		np.random.shuffle(haps_sampled)
		hap_A = haps_sampled[:num_samples]
		hap_B = haps_sampled[num_samples:]

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

		gram = self.haps_cov * self.snp_count

		causal_effects = np.random.normal(0, 1, num_causal)
		causal_snps = np.zeros(self.snp_count)
		causal_snps[causal_config.astype(bool)] = causal_effects
		# print(causal_snps) ####

		var_causal_raw = causal_snps.dot(gram.dot(causal_snps)) / self.snp_count
		scale = herit / var_causal_raw
		causal_snps_scaled = causal_snps * np.sqrt(scale)

		signal = gram.dot(causal_snps_scaled)
		# print(signal) ####
		noise = np.random.multivariate_normal(np.zeros(self.snp_count), gram * (1-herit))
		# print(noise) ####
		haps_var = np.diagonal(self.haps_cov)
		z_scores = (signal + noise) / np.sqrt(self.snp_count * haps_var * (1-herit))

		corr = self.haps_cov / np.sqrt(np.outer(haps_var, haps_var))
		corr = np.nan_to_num(corr)
		np.fill_diagonal(corr, 1.0)

		data_dict = {
			"z_gwas": z_scores,
			"ld_gwas": corr,
		}

		# raise Exception ####

		return data_dict
