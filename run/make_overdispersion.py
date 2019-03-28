from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import os

try:
	import cPickle as pickle
except ImportError:
	import pickle

def make_overdispersion(in_path, out_path):
	overdispersion = {}

	with open(in_path) as in_file:
		for line in in_file:
			data = line.split()
			sample = data[0]
			val = float(data[1])
			overdispersion[sample] = val

	with open(out_path, "wb") as out_file:
		pickle.dump(overdispersion, out_file)

if __name__ == '__main__':
	# Kidney Data
	in_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/overdispersion/KIRC.ALL.AS.CNV"
	out_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/overdispersion"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.path.join(out_dir, "overdispersion.pickle")

	make_overdispersion(in_path, out_path)

	# Prostate Data
	in_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq/overdispersion/calculated_overdispersion_values.txt"
	out_dir = "/bcb/agusevlab/awang/job_data/prostate_chipseq/overdispersion"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.path.join(out_dir, "overdispersion.pickle")

	make_overdispersion(in_path, out_path)
