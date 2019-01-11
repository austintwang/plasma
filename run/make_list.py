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
	in_dir = "/bcb/agusevlab/DATA/KIRC_RNASEQ/ASSOC"
	out_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists"

	# Kidney Data, Tumor
	in_path_tumor_01 = os.path.join(in_dir, "KIRC.T.FDR001.genes")
	out_path_tumor_01 = os.path.join(out_dir, "tumor_fdr001")

	in_path_tumor_5 = os.path.join(in_dir, "KIRC.T.FDR05.genes")
	out_path_tumor_5 = os.path.join(out_dir, "tumor_fdr05")

	make_list(in_path_tumor_01, out_path_tumor_01)
	make_list(in_path_tumor_5, out_path_tumor_5)

	# Kidney Data, Normal
	in_path_normal_01 = os.path.join(in_dir, "KIRC.N.FDR001.genes")
	out_path_normal_01 = os.path.join(out_dir, "normal_fdr001")

	in_path_normal_5 = os.path.join(in_dir, "KIRC.N.FDR05.genes")
	out_path_normal_5 = os.path.join(out_dir, "normal_fdr05")

	make_list(in_path_normal_01, out_path_normal_01)
	make_list(in_path_normal_5, out_path_normal_5)