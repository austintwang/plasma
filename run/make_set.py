import os
import pickle

def make_set(in_path, out_tumor_path, out_normal_path):
	tumor_set = set()
	normal_set = set()

	with open(in_path) as in_file:
		for line in in_file:
			name, definition = line.split()
			if definition == "0":
				tumor_set.add(name)
			else:
				normal_set.add(name)

	with open(out_tumor_path, "wb") as out_tumor_file:
		pickle.dump(tumor_set, out_tumor_file)

	with open(out_normal_path, "wb") as out_normal_file:
		pickle.dump(normal_set, out_normal_file)

if __name__ == '__main__':
	# Kidney Data
	in_path = "/bcb/agusevlab/DATA/KIRC_RNASEQ/ASVCF/KIRC.ALL.AS.PHE"
	out_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_tumor_path = os.path.join(out_dir, "tumor.pickle")
	out_normal_path = os.path.join(out_dir, "normal.pickle")

	make_set(in_path, out_tumor_path, out_normal_path)

	# Prostate Data
	in_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq/sample_sets/prostate_tn"
	out_dir = "/bcb/agusevlab/awang/job_data/prostate_chipseq/sample_sets"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_tumor_path = os.path.join(out_dir, "tumor.pickle")
	out_normal_path = os.path.join(out_dir, "normal.pickle")

	make_set(in_path, out_tumor_path, out_normal_path)
