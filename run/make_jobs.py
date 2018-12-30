from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import gzip
import sys
try:
	import cpickle as pickle
except ImportError:
	import pickle


def finalize(data, jobs_dir):
	# print("owiehofwieof") ####
	name = data["name"]
	target_path = os.path.join(jobs_dir, name)
	out_path = os.path.join(target_path, "input.pickle")
	if not os.path.isdir(target_path):
		os.makedirs(target_path)

	# # print(data["counts_total"]) ####

	# select = np.logical_and(data["counts1"] >= 1, data["counts2"] >= 1) 

	# # num_ppl_raw = np.size(data["counts1"])
	# # max_ppl = hyperparams.get("max_ppl")
	# # if max_ppl and max_ppl < num_ppl_raw:
	# # 	threshold = np.array([1] * max_ppl + [0] * (num_ppl_raw - max_ppl))
	# # 	np.random.shuffle(threshold)
	# # 	select = np.logical_and(select, threshold)
	# # 	data["num_ppl"] = max_ppl

	# data["num_snps_imbalance"] = len(data["hap1"])
	# data["num_snps_total_exp"] = data["num_snps_imbalance"]

	# data["hap1"] = np.stack(data["hap1"], axis=1)[select]
	# data["hap2"] = np.stack(data["hap2"], axis=1)[select]
	# data["counts1"] = data["counts1"][select]
	# data["counts2"] = data["counts2"][select]
	# data["counts_total"] = data["counts_total"][select]

	# num_ppl_raw = np.size(data["counts1"])
	# max_ppl = hyperparams.get("max_ppl")
	# if max_ppl and max_ppl < num_ppl_raw:
	# 	threshold = np.array([1] * max_ppl + [0] * (num_ppl_raw - max_ppl)).astype(np.bool)
	# 	# print(threshold) ####
	# 	np.random.shuffle(threshold)
	# 	# print(threshold) ####
	# 	# print(np.size(data["counts1"])) ####
	# 	data["hap1"] = data["hap1"][threshold]
	# 	data["hap2"] = data["hap2"][threshold]
	# 	data["counts1"] = data["counts1"][threshold]
	# 	data["counts2"] = data["counts2"][threshold]
	# 	data["counts_total"] = data["counts_total"][threshold]
	# 	# print(np.size(data["counts1"])) ####

	# data["num_ppl"] = np.size(data["counts1"])
	# # print(data["num_ppl"]) ####
	# # print(max_ppl) ####


	# # print(data["counts_total"]) ####
	# # print(name) ####
	
	# data.update(hyperparams)

	with open(out_path, "wb") as outfile:
		pickle.dump(data, outfile)
	return target_path

def make_targets(chr_paths, bed_path, out_dir, margin):
	# chr_paths = [os.path.join(chr_dir, i) for i in os.listdir(chr_dir)]
	jobs_dir = os.path.join(out_dir, "jobs")

	bed_info = []
	bed_start = False
	with open(bed_path) as bed_file:
		for line in bed_file:
			if not bed_start:
				if not line.startswith("chr"):
					continue
				bed_start = True
			entry = line.split()[0:4]
			bed_info.append(entry)

	target_data = [
		{
			"chr": str(i[0][3:]),
			"begin": int(i[1]) - margin,
			"end": int(i[2]) + margin,
			"name": i[3].strip("\""),
			"hap1": [],
			"hap2": [],
			"counts1": None,
			"counts2": None,
			"counts_total": None,
		} for i in bed_info
	]
	# print(target_data) ####

	active_ids = {}
	max_active = -1
	finish = False
	target_final = len(target_data) - 1

	for c in chr_paths:
		chr_start = False
		chr_idx = None
		pos_idx = None
		ppl_ids = []
		num_ppl = None
		gt_sidx = None
		as_sidx = None

		if c.endswith(".gz"):
			file_open = gzip.open
		else:
			file_open = open

		with file_open(c, "rb") as c_file:
			sys.stdout.write("{0}, {1}".format(max_active, c))
			for line in c_file:
				# print(max_active) ####
				# print(active_ids) ####
				# raw_input() ####
				# print(line) ####
				# input("") ####
				if not chr_start:
					if (not line.startswith("##")) and line.startswith("#"):
						# print(line) ####
						cols = line[1:].split()
						# print(cols) ####
						ppl_start = False
						for ind, col in enumerate(cols):
							if not ppl_start:
								if col == "CHROM":
									chr_idx = ind
								elif col == "POS":
									pos_idx = ind
								elif col == "FORMAT":
									fmt_idx = ind
									# fmt = col.split(":")
									# for sind, entry in enumerate(fmt):
									# 	if entry == "GT":
									# 		gt_sidx = sind
									# 		print(sind) ####
									# 	elif entry == "AS":
									# 		as_sidx = sind
									ppl_start = True
							else:
								ppl_ids.append(ind)
						num_ppl = len(ppl_ids)
						chr_start = True
						# print(ppl_ids) ####
						# print(num_ppl) ####
				
				else:
					cols = line.split()
					chr_num = str(cols[chr_idx][3:])
					pos = int(cols[pos_idx])
					# print(pos) ####
					fmt_str = cols[fmt_idx]
					fmt = fmt_str.split(":")
					# print(fmt) ####
					for sind, entry in enumerate(fmt):
						if entry == "GT":
							gt_sidx = sind
							# print(sind) ####
						elif entry == "AS":
							as_sidx = sind

					remove_ids = {}
					for k, v in active_ids.viewitems():
						if (v["chr"] != chr_num) or (v["end"] < pos):
							remove_ids[k] = v
					# print(remove_ids.keys()) ####
					for k, v in remove_ids.viewitems():
						path = finalize(v, jobs_dir)
						active_ids.pop(k, None)
						target_data[k] = path
					while True:
						# print(target_data[max_active+1]["chr"]) ####
						# if max_active not in active_ids:
						# 	break
						if max_active + 2 > len(target_data):
							break
						if target_data[max_active+1]["chr"] != chr_num:
							break
						if target_data[max_active+1]["begin"] > pos:
							break
						max_active += 1
						active_ids[max_active] = target_data[max_active]
						active_ids[max_active]["counts1"] = np.zeros(num_ppl)
						active_ids[max_active]["counts2"] = np.zeros(num_ppl)
						active_ids[max_active]["counts_total"] = np.zeros(num_ppl)
						# max_active += 1

					hap1_all = np.empty(num_ppl) 
					hap2_all = np.empty(num_ppl)
					counts1 = np.zeros(num_ppl)
					counts2 = np.zeros(num_ppl)
					counts_total = np.zeros(num_ppl)

					# print(ppl_ids) ####
					for ind, val in enumerate(ppl_ids):
						person = cols[val].split(":")
						gen_data = person[gt_sidx]
						read_data = person[as_sidx]
						haps = gen_data.split("|")
						hap1 = int(haps[0])
						hap2 = int(haps[1])
						hap1_all[ind] = hap1
						hap2_all[ind] = hap2

						reads = read_data.split(",")
						# print(reads) ####
						ref_reads = int(reads[0])
						alt_reads = int(reads[1])
						counts_total[ind] += ref_reads + alt_reads
						if hap1 == 0 and hap2 == 1:
							counts1[ind] += ref_reads
							counts2[ind] += alt_reads
						elif hap1 == 1 and hap2 == 0:
							counts2[ind] += ref_reads
							counts1[ind] += alt_reads

					for k, v in active_ids.viewitems():
						v["hap1"].append(hap1_all)
						v["hap2"].append(hap2_all)
						v["counts1"] += counts1
						v["counts2"] += counts2
						v["counts_total"] += counts_total

					# print(max_active, target_final) ####
					if max_active == target_final and len(active_ids) == 0:
						finish = True
						break
		if finish:
			# print("Done building data")
			break

	for k, v in active_ids.viewitems():
		path = finalize(v, jobs_dir)
		target_data[k] = path

	print("Done")



if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))

	# # Test Run
	# chr_dir_test = os.path.join(curr_path, "test_data", "chrs")
	# chr_paths = [chr_dir_test + "KIRC.ALL.AS.chr22.vcf"]
	# bed_path_test = os.path.join(curr_path, "test_data", "test_22.bed")
	# out_dir = os.path.join(curr_path, "test_results")
	# script_path = os.path.join(curr_path, "job.py")
	# # hyperparams = {
	# # 	"overdispersion": 0.05,
	# # 	"prop_noise_eqtl": 0.95,
	# # 	"prop_noise_ase": 0.50,
	# # 	"std_fraction": 0.75,
	# # 	"min_causal": 1,
	# # 	"num_causal": 1,
	# # 	"search_mode": "exhaustive",
	# # 	"max_causal": 1,
	# # 	"confidence": 0.95, 
	# # 	"max_ppl": 100
	# # }

	# make_targets(
	# 	chr_paths, 
	# 	bed_path_test, 
	# 	out_dir, 
	# 	30000, 
	# )

	# Kidney Data
	chr_dir = "/bcb/agusevlab/DATA/KIRC_RNASEQ/ASVCF"
	chrs = ["KIRC.ALL.AS.chr{0}.vcf.gz".format(i + 1) for i in xrange(22)]
	chr_paths = [os.path.join(chr_dir, i) for i in chrs]
	bed_path = "/bcb/agusevlab/DATA/KIRC_RNASEQ/ASVCF/gencode.protein_coding.transcripts.bed"
	out_dir = "/bcb/agusevlab/awang/job_data/kidney"

	make_targets(
		chr_paths, 
		bed_path, 
		out_dir, 
		30000, 
	)




