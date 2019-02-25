from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import sys
import string

try:
	import cPickle as pickle
except ImportError:
	import pickle

def write_bed(bed_data, output_name):
	non_numeric_chars = string.printable[10:]
	keys_sorted = sorted(bed_data.keys(), key=lambda x: bed_data[x]["start"])
	keys_sorted.sort(key=lambda x: int(bed_data[x]["chr"].translate(None, non_numeric_chars)))
	bed_list = [
		"chr{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
			bed_data[i]["chr"], 
			bed_data[i]["start"], 
			bed_data[i]["end"], 
			i, 
			bed_data[i]["ppa"], 
			bed_data[i]["gene"]
		) 
		for i in keys_sorted
	]
	with open(output_name, "w") as outfile:
		outfile.writelines(bed_list) 

def make_bed(input_path, output_path, model_flavors):
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	if model_flavors == "all":
		model_flavors = set(["full", "indep", "eqtl", "ase", "acav"])

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
				if "data_error" in result:
					continue
		except (EOFError, IOError):
			continue

		# print(result.keys()) ####
		# print(result["ldsr_data_indep"]) ####
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

	if "full" in model_flavors:
		write_bed(bed_data_all["full"], os.path.join(output_path, "ldsr_full.bed"))
	if "indep" in model_flavors:
		write_bed(bed_data_all["indep"], os.path.join(output_path, "ldsr_indep.bed"))
	if "eqtl" in model_flavors:
		write_bed(bed_data_all["eqtl"], os.path.join(output_path, "ldsr_eqtl.bed"))
	if "ase" in model_flavors:
		write_bed(bed_data_all["ase"], os.path.join(output_path, "ldsr_ase.bed"))
	if "acav" in model_flavors:
		write_bed(bed_data_all["acav"], os.path.join(output_path, "ldsr_acav.bed"))

if __name__ == '__main__':
	# # Kidney Data, Tumor
	# input_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"
	# output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_tumor_all"
	# model_flavors = "all"

	# make_bed(input_path, output_path, model_flavors)

	# # Kidney Data, Normal
	# input_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_all"
	# output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_normal_all"
	# model_flavors = "all"

	# make_bed(input_path, output_path, model_flavors)

	# Prostate Data, Tumor
	input_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_all"
	output_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_tumor_all"
	model_flavors = set(["indep", "eqtl", "ase", "acav"])

	make_bed(input_path, output_path, model_flavors)

	# Prostate Data, Normal
	input_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_all"
	output_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_normal_all"
	model_flavors = set(["indep", "eqtl", "ase", "acav"])

	make_bed(input_path, output_path, model_flavors)
