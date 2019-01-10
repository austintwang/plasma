#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import gzip
import sys
try:
	import cPickle as pickle
except ImportError:
	import pickle


def finalize(data, jobs_dir):
	# print("owiehofwieof") ####
	data["snp_ids"] = np.array(data["snp_ids"])
	data["snp_pos"] = np.array(data["snp_pos"])
	data["hap1"] = np.stack(data["hap1"], axis=1)
	data["hap2"] = np.stack(data["hap2"], axis=1)

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

def make_chr(chr_path, bed_path, out_dir, margin, chr_num):
	jobs_dir = os.path.join(out_dir, "jobs")

	bed_info = []
	# bed_start = False
	with open(bed_path) as bed_file:
		for line in bed_file:
			# if not bed_start:
			# 	if not line.startswith("chr"):
			# 		continue
			# 	bed_start = True
			if line.startswith("chr{0}".format(chr_num)):
				entry = line.split()[0:5]
				bed_info.append(entry)

	margin = int(margin)
	target_data = [
		{
			"chr": str(chr_num),
			"begin": int(i[4]) - margin,
			"end": int(i[4]) + margin,
			"tss": int(i[4]),
			"snp_ids": [],
			"snp_pos": [],
			"name": i[3].strip("\""),
			"hap1": [],
			"hap2": [],
			"counts1": None,
			"counts2": None,
			"counts_total": None,
		} for i in bed_info
	]
	# print(target_data) ####
	target_data.sort(key=lambda x: x["tss"])

	active_ids = {}
	max_active = -1
	# finish = False
	target_final = len(target_data) - 1

	chr_start = False
	# chr_idx = None
	pos_idx = None
	ppl_ids = []
	ppl_names = []
	num_ppl = None
	gt_sidx = None
	as_sidx = None

	if chr_path.endswith(".gz"):
		file_open = gzip.open
	else:
		file_open = open

	with file_open(chr_path, "rb") as c_file:
		sys.stdout.write("{0}, {1}".format(max_active, chr_path))
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
							if col == "ID":
								id_idx = ind
							elif col == "CHROM":
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
							ppl_names.append(col)
					num_ppl = len(ppl_ids)
					chr_start = True
					# print(ppl_ids) ####
					# print(num_ppl) ####
			
			else:
				cols = line.split()
				chr_num = cols[chr_idx][3:]
				pos = int(cols[pos_idx])
				# print(pos) ####
				if cols[id_idx] == ".":
					snp_id = "{0}.{1}".format(chr_num, pos)

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
					active_ids[max_active]["sample_names"] = ppl_names
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
					v["snp_ids"].append(snp_id)
					v["snp_pos"].append(pos)
					v["hap1"].append(hap1_all)
					v["hap2"].append(hap2_all)
					v["counts1"] += counts1
					v["counts2"] += counts2
					v["counts_total"] += counts_total

				# print(max_active, target_final) ####
				if max_active == target_final and len(active_ids) == 0:
					break

	for k, v in active_ids.viewitems():
		path = finalize(v, jobs_dir)
		target_data[k] = path

	print("Done")



if __name__ == '__main__':
	# curr_path = os.path.abspath(os.path.dirname(__file__))

	chr_path = sys.argv[1]
	bed_path = sys.argv[2]
	out_dir = sys.argv[3]
	margin = sys.argv[4]
	chr_num = sys.argv[5]
	make_chr(chr_path, bed_path, out_dir, margin, chr_num)

	