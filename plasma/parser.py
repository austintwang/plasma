import numpy as np 
from .reader import SubsetReader

class Universe(object):
	def __contains__(self, other):
		return True

def parse_locus(
	self, 
	vcf_path, 
	chrom, 
	gen_start, 
	gen_end,
	phe_start,
	phe_end,
	sample_filter=None,
	snp_filter=None,
	maf_thresh=0.
):
	if snp_filter is not None:
		snp_filter = set(snp_filter)
	else:
		snp_filter = Universe()

	if sample_filter is not None:
		sample_filter = set(sample_filter)
	else:
		sample_filter = Universe()	

	vcf_reader = SubsetReader(
		filename=vcf_path, 
		snp_subset=snp_filter, 
		sample_subset=sample_filter
	)

	num_samples = len(vcf_reader.samples)

	counts_total = np.zeros(num_samples)
	counts_A = np.zeros(num_samples)
	counts_B = np.zeros(num_samples)

	hap_A = []
	hap_B = []
	snp_ids = []
	snp_count = 0

	abs_start = min(gen_start, phe_start)
	abs_end = max(gen_end, phe_end)

	region = vcf_reader.fetch(chrom, abs_start - 1, abs_end)
	for record in region:
		chr_rec = record.CHROM
		pos = int(record.POS) + 1

		if record.ID == ".":
			snp_id = "{0}.{1}".format(chr_rec, pos)
		else:
			snp_id = record.ID

		if gen_start <= pos <= gen_end:
			hap_A_record = np.empty(num_samples)
			hap_B_record = np.empty(num_samples)
		
		if phe_start <= pos <= phe_end:
			counts_total_record = np.empty(num_samples)
			counts_A_record = np.empty(num_samples)
			counts_B_record = np.empty(num_samples)

		include_marker = True

		for ind, sample in enumerate(record.samples):
			gen_data = sample["GT"]
			if "/" in gen_data:
				include_marker = False
				break

			hap_data = gen_data.split("|")
			hap_A_sample = int(hap_data[0])
			hap_B_sample = int(hap_data[1])

			if gen_start <= pos <= gen_end:
				hap_A_record[ind] = hap_A_sample
				hap_B_record[ind] = hap_B_sample

			if phe_start <= pos <= phe_end:
				read_data = sample["AS"]

				ref_reads = int(read_data[0])
				alt_reads = int(read_data[1])
				counts_total_record[ind] = ref_reads + alt_reads

				if (hap_A_sample == 0) and (hap_B_sample == 1):
					counts_A_record[ind] = ref_reads
					counts_B_record[ind] = alt_reads
				elif (hap_A_sample == 1) and (hap_B_sample == 0):
					counts_B_record[ind] = ref_reads
					counts_A_record[ind] = alt_reads

		if not include_marker:
			continue

		freq = (np.mean(hap_A_record) + np.mean(hap_B_record)) / 2.
		maf = min(freq, 1 - freq)
		if maf < maf_thresh:
			continue

		if gen_start <= pos <= gen_end:
			hap_A.append(hap_A_record)
			hap_B.append(hap_B_record)
			snp_ids.append(snp_id)
			snp_count += 1

		if phe_start <= pos <= phe_end:
			counts_total += counts_total_record
			counts_A + counts_A_record
			counts_B + counts_B_record

	if snp_count == 0:
		raise ValueError("Specified region yielded no markers")

	hap_A = np.array(hap_A).T
	hap_B = np.array(hap_B).T

	snp_ids = np.array(snp_ids)
	snp_count = snp_count
	num_samples = np.shape(haps)[0]

	haps_means = np.mean(haps, axis=0)
	haps_centered = haps - haps_means
	haps_cov = np.nan_to_num(np.cov(haps_centered.T))

	data_dict = {
		"total_exp": counts_total,
		"counts_A": counts_A,
		"counts_B": counts_B,
		"hap_A": hap_A,
		"hap_B": hap_B,
		"haps_cov": haps_cov
	}

	return data_dict