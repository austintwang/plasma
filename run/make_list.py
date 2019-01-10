from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import sys

try:
	import cPickle as pickle
except ImportError:
	import pickle

make_list(in_path, out_path):
	with open(in_path) as in_file:
		gene_list = [line.rstrip() for line in in_file]

	with open(out_path, "wb") as out_file:
		pickle.dump(gene_list, out_file)

if __name__ == '__main__':
	out_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ_ASVCF/gene_lists"

	# Kidney Data, Tumor
	in_path_tumor_01 = ""
	out_path_tumor_01 = os.path.join(out_dir, "")

	in_path_tumor_5 = ""
	out_path_tumor_5 = os.path.join(out_dir, "")

	make_list(in_path_tumor_01, out_path_tumor_01)
	make_list(in_path_tumor_5, out_path_tumor_5)

	# Kidney Data, Normal