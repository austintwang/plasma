from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
try:
	import subprocess32 as subprocess
except ImportError:
	import subprocess
try:
	import cpickle as pickle
except ImportError:
	import pickle

LOCAL = False

def finalize(data, jobs_dir, hyperparams):
	# print("owiehofwieof") ####
	name = data["name"]
	target_path = os.path.join(jobs_dir, name)
	out_path = os.path.join(target_path, "input.pickle")
	if not os.path.isdir(target_path):
		os.makedirs(target_path)

	# print(data["counts_total"]) ####

	select = np.logical_and(data["counts1"] >= 1, data["counts2"] >= 1) 

	# num_ppl_raw = np.size(data["counts1"])
	# max_ppl = hyperparams.get("max_ppl")
	# if max_ppl and max_ppl < num_ppl_raw:
	# 	threshold = np.array([1] * max_ppl + [0] * (num_ppl_raw - max_ppl))
	# 	np.random.shuffle(threshold)
	# 	select = np.logical_and(select, threshold)
	# 	data["num_ppl"] = max_ppl

	data["num_snps_imbalance"] = len(data["hap1"])
	data["num_snps_total_exp"] = data["num_snps_imbalance"]

	data["hap1"] = np.stack(data["hap1"], axis=1)[select]
	data["hap2"] = np.stack(data["hap2"], axis=1)[select]
	data["counts1"] = data["counts1"][select]
	data["counts2"] = data["counts2"][select]
	data["counts_total"] = data["counts_total"][select]

	num_ppl_raw = np.size(data["counts1"])
	max_ppl = hyperparams.get("max_ppl")
	if max_ppl and max_ppl < num_ppl_raw:
		threshold = np.array([1] * max_ppl + [0] * (num_ppl_raw - max_ppl)).astype(np.bool)
		# print(threshold) ####
		np.random.shuffle(threshold)
		# print(threshold) ####
		# print(np.size(data["counts1"])) ####
		data["hap1"] = data["hap1"][threshold]
		data["hap2"] = data["hap2"][threshold]
		data["counts1"] = data["counts1"][threshold]
		data["counts2"] = data["counts2"][threshold]
		data["counts_total"] = data["counts_total"][threshold]
		# print(np.size(data["counts1"])) ####

	data["num_ppl"] = np.size(data["counts1"])
	# print(data["num_ppl"]) ####
	# print(max_ppl) ####


	# print(data["counts_total"]) ####
	# print(name) ####
	
	data.update(hyperparams)

	with open(out_path, "wb") as outfile:
		pickle.dump(data, outfile)
	return target_path


def make_targets(chr_paths, bed_path, jobs_dir, margin, hyperparams):
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

		with open(c) as c_file:
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
						path = finalize(v, jobs_dir, hyperparams)
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
			print("Done building data")
			break

def plot(result, out_dir, name):
	set_sizes_full = result["set_sizes_full"]
	set_sizes_indep = result["set_sizes_indep"]
	set_sizes_eqtl = result["set_sizes_eqtl"]
	set_sizes_ase = result["set_sizes_ase"]
	set_sizes_caviar_ase = result["set_sizes_caviar_ase"]

	try:
		sns.set(style="white")
		sns.distplot(
			set_sizes_full,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="Full"
		)
		sns.distplot(
			set_sizes_indep,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="Independent Likelihoods"
		)
		sns.distplot(
			set_sizes_eqtl,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="eQTL-Only"
		)
		sns.distplot(
			set_sizes_ase,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="ASE-Only"
		)
		sns.distplot(
			set_sizes_caviar_ase,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="CAVIAR-ASE"
		)
		plt.xlim(0, None)
		plt.legend(title="Model")
		plt.xlabel("Set Size")
		plt.ylabel("Density")
		plt.title("Distribution of Causal Set Sizes: {0}".format(name))
		plt.savefig(os.path.join(out_dir, "set_size_distribution.svg"))
		plt.clf()
	except Exception:
		plt.clf()

def interpret(targets, out_dir, jobs_dir, name):
	summary = {
		"set_sizes_full": [],
		"set_sizes_indep": [],
		"set_sizes_eqtl": [],
		"set_sizes_ase": [],
		"set_sizes_caviar_ase": []
	}

	for t in targets:
		result_path = os.path.join(jobs_dir, t, "output.pickle")
		with open(result_path, "wb") as result_file:
			result = pickle.dump(result_file)

		summary["set_sizes_full"].append(np.size(result["causal_set_full"])) 
		summary["set_sizes_indep"].append(np.size(result["causal_set_indep"]))
		summary["set_sizes_eqtl"].append(np.size(result["causal_set_eqtl"])) 
		summary["set_sizes_ase"].append(np.size(result["causal_set_ase"])) 
		summary["set_sizes_caviar_ase"].append(np.size(result["causal_set_caviar_ase"])) 

	plot(summary, out_dir, name)

	return summary

def dispatch(target_dir, script_path):
	stdout_path = os.path.join(target_dir, "stdout.txt")
	stderr_path = os.path.join(target_dir, "stderr.txt")
	# print("dispatch") ####

	try:
		args = ["qsub", "-e", stderr_path, "-o", stdout_path, "-v", "DATA_DIR=\""+target_dir+"\"", script_path]
		# args = [script_path, target_dir] ####
		# print(args) ####
		job_info = subprocess.check_output(args)
		# print(job_info) ####
		if LOCAL:
			job_id = job_info.rstrip()
		else:
			job_id = job_info.split()[2]

		return job_id
	except subprocess.CalledProcessError as e:
		raise e
		return False

def poll(job_id):
	args = ["qstat", "-f", job_id]
	job_info = subprocess.check_output(args)
	# print(job_info) ####
	exit_code = None
	lines = job_info.split("\n")
	for l in lines:
		stat = l.strip().split(" = ")
		if stat[0] == "job_state":
			state = stat[1]
		if stat[0] == "exit_status":
			exit_code = stat[1]

	print(job_id, state, exit_code) ####
	return state, exit_code


	# if len(lines) > 2:
	# 	header = [i.lower() for i in lines[0]]
	# 	if LOCAL:
	# 		state_idx = header.index("s")
	# 	else:
	# 		state_idx = header.index("state")
	# 	job_state = lines[2][state_idx]
	# 	# print(lines[2]) ####
	# 	raise Exception ####
	# 	return job_state
	# raise Exception
	# return None

def delete(job_id):
	args = ["qdel", job_id]

def run(chr_dir, bed_path, out_dir, margin, hyperparams, num_tasks, poll_freq, script_path, name, parse_input=True):
	chr_paths = [os.path.join(chr_dir, i) for i in os.listdir(chr_dir)]
	jobs_dir = os.path.join(out_dir, "jobs")
	if parse_input:
		make_targets(chr_paths, bed_path, jobs_dir, margin, hyperparams)
	targets = os.listdir(jobs_dir)

	wait_pool = set(targets)
	active_pool = {}
	complete_pool = set()
	fail_pool = set()
	dead_pool = set()
	# print(wait_pool) ####
	try:
		while (len(wait_pool) > 0) or (len(active_pool) > 0):
			# print("woiehoifwe") ####
			to_remove = set()
			for k, v in active_pool.viewitems():
				state, exit_code = poll(v)
				# print(state) ####
				if not LOCAL and "E" in state:
					delete(v)
					fail_pool.add(k)
					to_remove.add(k)
					# active_pool.pop(k)
				elif exit_code and exit_code != 0:
					# print(exit_code) ####
					fail_pool.add(k)
					to_remove.add(k)
				elif "C" in state or state is None:
					complete_pool.add(k)
					to_remove.add(k)
					# active_pool.pop(k)
			for i in to_remove:
				active_pool.pop(i)

			vacant = num_tasks - len(active_pool)
			for _ in xrange(vacant):
				if len(wait_pool) == 0:
					break
				target = wait_pool.pop()
				job_id = dispatch(target, script_path)
				active_pool[target] = job_id

			time.sleep(poll_freq)

		while (len(fail_pool) > 0) or (len(active_pool) > 0):
			to_remove = set()
			for k, v in active_pool.viewitems():
				state, exit_code = poll(v)
				if not LOCAL and "E" in state:
					delete(v)
					dead_pool.add(k)
					to_remove.add(k)
				elif exit_code and exit_code != 0:
					# print(exit_code) ####
					dead_pool.add(k)
					to_remove.add(k)
				elif "C" in state or state is None:
					complete_pool.add(k)
					to_remove.add(k)

			for i in to_remove:
				active_pool.pop(i)

			vacant = num_tasks - len(active_pool)
			for _ in xrange(vacant):
				if len(fail_pool) == 0:
					break
				target = fail_pool.pop()
				job_id = dispatch(target, script_path)
				active_pool[target] = job_id

			time.sleep(poll_freq)

	finally:
		for v in active_pool.values():
			delete(v)

	return interpret(complete_pool, out_dir, jobs_dir, name)


if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))

	# Test Run
	chr_dir_test = os.path.join(curr_path, "test_data", "chrs")
	bed_path_test = os.path.join(curr_path, "test_data", "test_22.bed")
	out_dir = os.path.join(curr_path, "test_results")
	script_path = os.path.join(curr_path, "job.py")
	hyperparams = {
		"overdispersion": 0.05,
		"prop_noise_eqtl": 0.95,
		"prop_noise_ase": 0.50,
		"std_fraction": 0.75,
		"min_causal": 1,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"confidence": 0.95, 
		"max_ppl": 100
	}

	run(
		chr_dir_test, 
		bed_path_test, 
		out_dir, 
		30000, 
		hyperparams, 
		7, 
		1, 
		script_path, 
		"test_run",
		parse_input=False
	)





