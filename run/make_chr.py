#!/usr/bin/env python

import numpy as np
import string
import os
import gzip
import sys
import pysam
import vcf
import pickle


def finalize(data, jobs_dir):
	data["snp_ids"] = np.array(data["snp_ids"])
	data["snp_pos"] = np.array(data["snp_pos"])
	if len(data["hap1"]) == 0:
		data["hap1"] = np.empty([data["num_ppl"], 0])
	else:
		data["hap1"] = np.stack(data["hap1"], axis=1)
	if len(data["hap2"]) == 0:
		data["hap2"] = np.empty([data["num_ppl"], 0])
	else:
		data["hap2"] = np.stack(data["hap2"], axis=1)

	name = data["name"]
	target_path = os.path.join(jobs_dir, name)
	out_path = os.path.join(target_path, "input.pickle")
	if not os.path.isdir(target_path):
		os.makedirs(target_path)

	# print(data) ####
	with open(out_path, "wb") as outfile:
		pickle.dump(data, outfile)
	return target_path

def make_chr(chr_path, bed_path, out_dir, margin, chr_spec):
	jobs_dir = os.path.join(out_dir, "jobs")

	bed_info = []
	# bed_start = False
	with open(bed_path) as bed_file:
		for line in bed_file:
			# if not bed_start:
			# 	if not line.startswith("chr"):
			# 		continue
			# 	bed_start = True
			if chr_spec == "all" or line.startswith(chr_spec + "\t"):
				entry = line.split()
				bed_info.append(entry)

	target_data = []
	for i in bed_info:
		chr_num = i[0]
		margin = int(margin)
		gene_begin = int(i[1])
		gene_end = int(i[2])
		if len(entry) >= 5:
			center = int(i[4])
		else:
			center = (gene_begin + gene_end) // 2
		snps_begin = center - margin
		snps_end = center + margin
		
		abs_begin = min(snps_begin, gene_begin)
		abs_end = max(snps_end, gene_end)

		job_info = {
			"chr": str(chr_num),
			"center": center,
			"snps_begin": snps_begin,
			"snps_end": snps_end,
			"gene_begin": gene_begin,
			"gene_end": gene_end,
			"abs_begin": abs_begin,
			"abs_end": abs_end,
			"snp_ids": [],
			"snp_pos": [],
			"name": i[3].strip("\""),
			"hap1": [],
			"hap2": [],
			"counts1": None,
			"counts2": None,
			"counts_total": None,
		}
		target_data.append(job_info)

	# print(target_data) ####
	non_numeric_chars = string.printable[10:]

	target_data.sort(key=lambda x: x["abs_begin"])
	target_data.sort(key=lambda x: int((x["chr"]).translate(None, non_numeric_chars)))
	# print(target_data) ####

	active_ids = {}
	max_active = -1
	# finish = False
	target_final = len(target_data) - 1

	# pysam.tabix_index(chr_path, preset="vcf")

	vcf_reader = vcf.Reader(filename=chr_path)
	ppl_names = vcf_reader.samples
	num_ppl = len(ppl_names)

	# print(next(vcf_reader)) ####
	if chr_spec == "all":
		vcf_iter = vcf_reader
	else:
		vcf_iter = vcf_reader.fetch(chr_spec)

	for record in vcf_iter:
		chr_num = record.CHROM
		# print(chr_num) ####
		print(record) ####
		pos = int(record.POS) + 1
		# print(pos) ####
		if record.ID == ".":
			snp_id = "{0}.{1}".format(chr_num, pos)
		else:
			snp_id = record.ID

		remove_ids = {}
		for k, v in active_ids.items():
			if (v["chr"] != chr_num) or (v["abs_end"] < pos):
				remove_ids[k] = v
		# print(remove_ids.keys()) ####
		for k, v in remove_ids.items():
			path = finalize(v, jobs_dir)
			active_ids.pop(k, None)
			target_data[k] = path
		while True:
			# print(target_data[max_active+1]["chr"]) ####
			# if max_active not in active_ids:
			# 	break
			if max_active + 1 > len(target_data) - 1:
				break
			if target_data[max_active+1]["chr"] != chr_num:
				break
			if target_data[max_active+1]["abs_begin"] > pos:
				break
			max_active += 1
			active_ids[max_active] = target_data[max_active]
			active_ids[max_active]["counts1"] = np.zeros(num_ppl)
			active_ids[max_active]["counts2"] = np.zeros(num_ppl)
			active_ids[max_active]["counts_total"] = np.zeros(num_ppl)
			active_ids[max_active]["sample_names"] = ppl_names
			active_ids[max_active]["num_ppl"] = num_ppl
			# max_active += 1

		hap1_all = np.empty(num_ppl) 
		hap2_all = np.empty(num_ppl)
		counts1 = np.zeros(num_ppl)
		counts2 = np.zeros(num_ppl)
		counts_total = np.zeros(num_ppl)

		include_marker = True
		for ind, sample in enumerate(record.samples):
			gen_data = sample["GT"]
			
			if gen_data == "./.":
				include_marker = False
				break

			read_data = sample["AS"]

			ref_reads = int(read_data[0])
			alt_reads = int(read_data[1])
			counts_total[ind] += ref_reads + alt_reads

			haps = gen_data.split("|")
			hap1 = int(haps[0])
			hap2 = int(haps[1])
			hap1_all[ind] = hap1
			hap2_all[ind] = hap2

			if (hap1 == 0) and (hap2 == 1):
				counts1[ind] += ref_reads
				counts2[ind] += alt_reads
			elif (hap1 == 1) and (hap2 == 0):
				counts2[ind] += ref_reads
				counts1[ind] += alt_reads

		if include_marker:
			for k, v in active_ids.items():
				if v["snps_begin"] <= pos < v["snps_end"]:
					v["snp_ids"].append(snp_id)
					v["snp_pos"].append(pos)
					v["hap1"].append(hap1_all)
					v["hap2"].append(hap2_all)
				if v["gene_begin"] <= pos < v["gene_end"]:
					v["counts1"] += counts1
					v["counts2"] += counts2
					v["counts_total"] += counts_total

		# print(max_active, target_final) ####
		if (max_active == target_final) and (len(active_ids) == 0):
			break

	for k, v in active_ids.items():
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

	