from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import os

try:
	import cPickle as pickle
except ImportError:
	import pickle

def add_to_filter(snp_filter, in_path):
	with open(in_path) as bim_file:
		for line in bim_file:
			entry = line.split()
			snp_filter.add(entry[1])

def make_filter(in_dir, out_dir, name):
	bims = [i for i in os.listdir(in_dir) if i.endswith(".bim")]

	snp_filter = set()
	for b in bims:
		add_to_filter(snp_filter, b)

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.join(out_dir, name)

	with open(out_path, "wb") as out_file:
		pickle.dump(snp_filter, out_file)

if __name__ == '__main__':

	in_path = "/bcb/agusevlab/awang/job_data/sim_coloc/1000g/LDREF/"
	out_dir = "/bcb/agusevlab/awang/job_data/sim_coloc/1000g/"
	name = "snp_filter.pickle"

	make_filter(in_path, out_dir, name)