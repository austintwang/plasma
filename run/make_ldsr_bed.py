import numpy as np
import os
import sys
import string
import pickle

def write_bed(bed_data, output_name):
	non_numeric_chars = string.printable[10:]
	keys_sorted = sorted(list(bed_data.keys()), key=lambda x: bed_data[x]["start"])
	keys_sorted.sort(key=lambda x: int(bed_data[x]["chr"].translate(None, non_numeric_chars)))
	# bed_list = [
	# 	"{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
	# 		bed_data[i]["chr"], 
	# 		bed_data[i]["start"], 
	# 		bed_data[i]["end"], 
	# 		i, 
	# 		bed_data[i]["ppa"], 
	# 		bed_data[i]["gene"]
	# 	) 
	# 	for i in keys_sorted
	# 	if bed_data[i]["ppa"] != np.nan
	# ]
	bed_list = []
	for i in keys_sorted:
		if bed_data[i]["ppa"] != np.nan:
			if str(bed_data[i]["chr"]).startswith("chr"):
				chr_data = bed_data[i]["chr"]
			else:
				chr_data = "chr" + str(bed_data[i]["chr"])
			entry = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
				chr_data, 
				bed_data[i]["start"], 
				bed_data[i]["end"], 
				i, 
				bed_data[i]["ppa"], 
				bed_data[i]["gene"]
			) 
			bed_list.append(entry)


	with open(output_name, "w") as outfile:
		outfile.writelines(bed_list) 

def make_bed(input_path, output_path, model_flavors):
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	if model_flavors == "all":
		model_flavors = set(["full", "indep", "eqtl", "ase", "acav", "fmb"])

	bed_data_all = {}
	bed_data_all["ctrl"] = {}
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
	if "fmb" in model_flavors:
		bed_data_all["fmb"] = {}

	targets = os.listdir(input_path)
	for t in targets:
		result_path = os.path.join(input_path, t, "output.pickle")
		try:
			with open(result_path, "rb") as result_file:
				result = pickle.load(result_file, encoding='latin1')
				if "data_error" in result:
					continue

				if "ldsr_data_fmb" not in result:
					continue
		except (EOFError, IOError):
			continue

		try:
			for k, v in result["bed_ctrl"].items():
					bed_data_all["ctrl"][k] = v
		except Exception:
			print(result_path) ####
			print(t) ####
			raise Exception ####

		# print(result.keys()) ####
		# print(result["ldsr_data_indep"]) ####
		if "full" in model_flavors:
			for k, v in result["ldsr_data_full"].items():
				if k in bed_data_all["full"]:
					bed_data_all["full"][k] = max(v, bed_data_all["full"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["full"][k] = v

		if "indep" in model_flavors:
			for k, v in result["ldsr_data_indep"].items():
				if k in bed_data_all["indep"]:
					bed_data_all["indep"][k] = max(v, bed_data_all["indep"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["indep"][k] = v

		if "eqtl" in model_flavors:
			for k, v in result["ldsr_data_eqtl"].items():
				if k in bed_data_all["eqtl"]:
					bed_data_all["eqtl"][k] = max(v, bed_data_all["eqtl"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["eqtl"][k] = v

		if "ase" in model_flavors:
			for k, v in result["ldsr_data_ase"].items():
				if k in bed_data_all["ase"]:
					bed_data_all["ase"][k] = max(v, bed_data_all["ase"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["ase"][k] = v

		if "acav" in model_flavors:
			for k, v in result["ldsr_data_acav"].items():
				if k in bed_data_all["acav"]:
					bed_data_all["acav"][k] = max(v, bed_data_all["acav"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["acav"][k] = v

		if "fmb" in model_flavors:
			for k, v in result["ldsr_data_fmb"].items():
				if k in bed_data_all["fmb"]:
					bed_data_all["fmb"][k] = max(v, bed_data_all["fmb"][k], key=lambda x: x["ppa"]) 
				else:
					bed_data_all["fmb"][k] = v

	write_bed(bed_data_all["ctrl"], os.path.join(output_path, "ctrl.bed"))
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
	if "fmb" in model_flavors:
		write_bed(bed_data_all["fmb"], os.path.join(output_path, "ldsr_acav.bed"))

if __name__ == '__main__':
	# Kidney Data, Tumor
	input_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"
	output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_tumor_all"
	model_flavors = set(["indep", "fmb", "ase", "acav"])

	make_bed(input_path, output_path, model_flavors)

	# Kidney Data, Normal
	input_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_all"
	output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_normal_all"
	model_flavors = set(["indep", "fmb", "ase", "acav"])

	make_bed(input_path, output_path, model_flavors)

	# Prostate Data, Tumor
	input_path = "/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_all"
	output_path = "/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_tumor_all"
	model_flavors = set(["indep", "fmb", "ase", "acav"])

	make_bed(input_path, output_path, model_flavors)

	# Prostate Data, Normal
	input_path = "/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_all"
	output_path = "/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_normal_all"
	model_flavors = set(["indep", "fmb", "ase", "acav"])

	make_bed(input_path, output_path, model_flavors)
