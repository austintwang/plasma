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

def write_bed(bed_data, output_name):
	keys_sorted = sorted(bed_data.keys(), key=lambda x: bed_data[x]["start"])
	keys_sorted.sort(bed_data.keys(), key=lambda x: int(bed_data[x]["chr"]))
	bed_list = [
		"chr{0}\t{1}\t{2}\t{3}\t{4}".format(i["chr"], i["start"], i["end"], i["ppa"], i["gene"]) 
		for i in keys_sorted
	]
	with open(output_name, "w") as outfile:
		outfile.writelines(bed_list) 

def make_bed(input_path, output_path, model_flavors):
	bed_data_all = {}
	if "full" in model_flavors:
		bed_data_all["full"] = {}
	if "indep" in model_flavors:
		bed_data_all["indep"] = {}
	if "eqtl" in model_flavors:
		bed_data_all["eqtl"] = {}
	if "ase" in model_flavors:
		bed_data_all["ase"] = {}
	if "acav" in model_flavors:
		bed_data_all["acav"] = {}

	targets = os.listdir(input_path)
	for t in targets:
		result_path = os.path.join(input_path, t, "output.pickle")
		try:
			with open(result_path, "rb") as result_file:
				result = pickle.load(result_file)
			except (EOFError, IOError):
				continue

		if "full" in model_flavors:
			for k, v in result["ldsr_data_full"].viewitems():
				if k in bed_data_all["full"]:
					bed_data_all["full"][k] = max(v, bed_data_all["full"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["full"][k] = v

		if "indep" in model_flavors:
			for k, v in result["ldsr_data_indep"].viewitems():
				if k in bed_data_all["indep"]:
					bed_data_all["indep"][k] = max(v, bed_data_all["indep"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["indep"][k] = v

		if "eqtl" in model_flavors:
			for k, v in result["ldsr_data_eqtl"].viewitems():
				if k in bed_data_all["eqtl"]:
					bed_data_all["eqtl"][k] = max(v, bed_data_all["eqtl"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["eqtl"][k] = v

		if "ase" in model_flavors:
			for k, v in result["ldsr_data_ase"].viewitems():
				if k in bed_data_all["ase"]:
					bed_data_all["ase"][k] = max(v, bed_data_all["ase"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["ase"][k] = v

		if "acav" in model_flavors:
			for k, v in result["ldsr_data_caviar_ase"].viewitems():
				if k in bed_data_all["acav"]:
					bed_data_all["acav"][k] = max(v, bed_data_all["acav"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["acav"][k] = v





def make_list(in_path, out_path):
	with open(in_path) as in_file:
		gene_list = [line.rstrip() for line in in_file]

	with open(out_path, "wb") as out_file:
		pickle.dump(gene_list, out_file)

if __name__ == '__main__':
	in_dir = "/bcb/agusevlab/DATA/KIRC_RNASEQ/ASSOC"
	out_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# Kidney Data, Tumor
	in_path_tumor_01 = os.path.join(in_dir, "KIRC.T.FDR001.genes")
	out_path_tumor_01 = os.path.join(out_dir, "tumor_fdr001.pickle")

	in_path_tumor_5 = os.path.join(in_dir, "KIRC.T.FDR05.genes")
	out_path_tumor_5 = os.path.join(out_dir, "tumor_fdr05.pickle")

	make_list(in_path_tumor_01, out_path_tumor_01)
	make_list(in_path_tumor_5, out_path_tumor_5)

	# Kidney Data, Normal
	in_path_normal_01 = os.path.join(in_dir, "KIRC.N.FDR001.genes")
	out_path_normal_01 = os.path.join(out_dir, "normal_fdr001.pickle")

	in_path_normal_5 = os.path.join(in_dir, "KIRC.N.FDR05.genes")
	out_path_normal_5 = os.path.join(out_dir, "normal_fdr05.pickle")

	make_list(in_path_normal_01, out_path_normal_01)
	make_list(in_path_normal_5, out_path_normal_5)