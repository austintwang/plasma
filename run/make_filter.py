import os
import gzip
import pickle

def make_filter(in_path, out_path):
	snp_filter = set()
	with gzip.open(in_path) as bed_file:
		for line in bed_file:
			entry = line.decode('UTF-8').split()
			snp_filter.add(entry[3])

	with open(out_path, "wb") as out_file:
		pickle.dump(snp_filter, out_file)

if __name__ == '__main__':

	# Prostate Data
	in_path = "/agusevlab/awang/job_data/prostate_chipseq/snp_filters/1KG_SNPs_filt.bed.gz"
	out_dir = "/agusevlab/awang/job_data/prostate_chipseq/snp_filters/"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.path.join(out_dir, "1KG_SNPs.pickle")

	make_filter(in_path, out_path)

	# Kidney Data
	in_path = "/agusevlab/awang/job_data/prostate_chipseq/snp_filters/1KG_SNPs_filt.bed.gz"
	out_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/snp_filters/"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.path.join(out_dir, "1KG_SNPs.pickle")

	make_filter(in_path, out_path)
