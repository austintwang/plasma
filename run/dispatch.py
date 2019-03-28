from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import time
import drmaa
# try:
# 	import subprocess32 as subprocess
# except ImportError:
# 	import subprocess
try:
	import cPickle as pickle
except ImportError:
	import pickle

# LOCAL = False

def dispatch(
		s, 
		target, 
		output_path, 
		input_path, 
		params_path, 
		script_path, 
		selection_path, 
		filter_path, 
		overdispersion_path
	):

	job_input_path = os.path.join(input_path, target, "input.pickle")
	job_output_path = os.path.join(output_path, target)

	if not os.path.exists(job_output_path):
		os.makedirs(job_output_path)

	stdout_path = ":" + os.path.join(job_output_path, "stdout.txt")
	stderr_path = ":" + os.path.join(job_output_path, "stderr.txt")

	environ = os.environ
	environ["SGE_O_SHELL"] = "/bcb/agusevlab/awang/python/bin/python"

	jt = s.createJobTemplate()
	jt.remoteCommand = script_path
	jt.args = [job_output_path, job_input_path, params_path, selection_path, filter_path, overdispersion_path]
	# jt.joinFiles = True
	jt.outputPath = stdout_path
	jt.errorPath = stderr_path
	jt.jobEnvironment = environ

	job_id = s.runJob(jt)

	s.deleteJobTemplate(jt)

	return job_id


	# print("dispatch") ####

	# try:
	# 	args = ["qsub", "-e", stderr_path, "-o", stdout_path, "-v", "DATA_DIR=\""+target_dir+"\"", script_path]
	# 	# args = [script_path, target_dir] ####
	# 	# print(args) ####
	# 	job_info = subprocess.check_output(args)
	# 	# print(job_info) ####
	# 	if LOCAL:
	# 		job_id = job_info.rstrip()
	# 	else:
	# 		job_id = job_info.split()[2]

	# 	return job_id
	# except subprocess.CalledProcessError as e:
	# 	raise e
	# 	return False

def poll(s, job_id):
	status = s.jobStatus(job_id)
	if status in [drmaa.JobState.DONE, drmaa.JobState.UNDETERMINED]:
		return "complete"
	elif status == drmaa.JobState.FAILED:
		return "failed"
	return "in_progress"

	# args = ["qstat", "-f", job_id]
	# job_info = subprocess.check_output(args)
	# # print(job_info) ####
	# exit_code = None
	# lines = job_info.split("\n")
	# for l in lines:
	# 	stat = l.strip().split(" = ")
	# 	if stat[0] == "job_state":
	# 		state = stat[1]
	# 	if stat[0] == "exit_status":
	# 		exit_code = stat[1]

	# print(job_id, state, exit_code) ####
	# return state, exit_code


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

def delete(s, job_id):
	s.control(job_id, drmaa.JobControlAction.TERMINATE)
	
	# args = ["qdel", job_id]

def run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path, 
		selection_path, 
		list_path, 
		filter_path, 
		overdispersion_path,
		params_name
	):
	if not os.path.exists(params_path):
		os.makedirs(params_path)

	hyperparams_path = os.path.join(params_path, params_name)

	with open(hyperparams_path, "wb") as params_file:
		pickle.dump(hyperparams, params_file)

	if list_path == "all":
		targets = os.listdir(input_path)
	else:
		with open(list_path, "rb") as list_file:
			targets = pickle.load(list_file)

	with drmaa.Session() as s:
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
					state = poll(s, v)
					# print(state) ####
					if state == "failed":
						delete(s, v)
						fail_pool.add(k)
						to_remove.add(k)
						# active_pool.pop(k)
					# elif exit_code and exit_code != 0:
					# 	# print(exit_code) ####
					# 	fail_pool.add(k)
					# 	to_remove.add(k)
					elif state == "complete":
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
					job_id = dispatch(
						s, 
						target, 
						output_path, 
						input_path, 
						hyperparams_path, 
						script_path, 
						selection_path,
						filter_path
					)
					active_pool[target] = job_id

				time.sleep(poll_freq)

			while (len(fail_pool) > 0) or (len(active_pool) > 0):
				to_remove = set()
				for k, v in active_pool.viewitems():
					state = poll(s, v)
					if state == "failed":
						delete(s, v)
						dead_pool.add(k)
						to_remove.add(k)
					# elif exit_code and exit_code != 0:
					# 	# print(exit_code) ####
					# 	dead_pool.add(k)
					# 	to_remove.add(k)
					elif state == "complete":
						complete_pool.add(k)
						to_remove.add(k)

				for i in to_remove:
					active_pool.pop(i)

				vacant = num_tasks - len(active_pool)
				for _ in xrange(vacant):
					if len(fail_pool) == 0:
						break
					target = fail_pool.pop()
					job_id = dispatch(
						s, 
						target, 
						output_path, 
						input_path, 
						params_path, 
						script_path, 
						selection_path,
						filter_path
					)
					active_pool[target] = job_id

				time.sleep(poll_freq)

		finally:
			for v in active_pool.values():
				delete(s, v)



if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))

	# # Test Run
	# out_dir = os.path.join(curr_path, "test_results")
	# script_path = os.path.join(curr_path, "job.py")
	# hyperparams = {
	# 	"overdispersion": 0.05,
	# 	"prop_noise_eqtl": 0.95,
	# 	"prop_noise_ase": 0.50,
	# 	"std_fraction": 0.75,
	# 	"min_causal": 1,
	# 	"num_causal": 1,
	# 	"coverage": 100,
	# 	"search_mode": "exhaustive",
	# 	"max_causal": 1,
	# 	"confidence": 0.95, 
	# 	"max_ppl": 100
	# }

	# run(
	# 	out_dir, 
	# 	30000, 
	# 	hyperparams, 
	# 	7, 
	# 	1, 
	# 	script_path, 
	# 	"test_run",

	# 	parse_input=False
	# )

	# Kidney Data, 1 CV
	input_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/jobs"
	params_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/params"
	script_path = os.path.join(curr_path, "job.py")
	overdispersion_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/overdispersion/overdispersion.pickle"

	hyperparams = {
		"prop_noise_eqtl": 0.95,
		"prop_noise_ase": 0.50,
		"min_causal": 1,
		"num_causal": 1.1,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"confidence": 0.95, 
		"model_flavors": "all"
	}

	num_tasks = 100
	poll_freq = 5

	# Normal
	list_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/normal_fdr05.pickle"
	selection_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/normal.pickle"

	# Normal, all samples
	params_name = "1cv_all.pickle"
	hyperparams["max_ppl"] = None
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_all"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		overdispersion_path,
		params_name
	)

	# Normal, 50 samples
	params_name = "1cv_50.pickle"
	hyperparams["max_ppl"] = 50
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_50"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		overdispersion_path,
		params_name
	)

	# Normal, 10 samples
	params_name = "1cv_10.pickle"
	hyperparams["max_ppl"] = 10
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_10"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		overdispersion_path,
		params_name
	)

	# Tumor
	list_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/tumor_fdr05.pickle"
	selection_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/tumor.pickle"

	# Tumor, all samples
	params_name = "1cv_all.pickle"
	hyperparams["max_ppl"] = None
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		overdispersion_path,
		params_name
	)

	# Tumor, 200 samples
	params_name = "1cv_200.pickle"
	hyperparams["max_ppl"] = 200
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_200"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		overdispersion_path,
		params_name
	)

	# Tumor, 100 samples
	params_name = "1cv_100.pickle"
	hyperparams["max_ppl"] = 100
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_100"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		overdispersion_path,
		params_name
	)

	# Tumor, 50 samples
	params_name = "1cv_50.pickle"
	hyperparams["max_ppl"] = 50
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_50"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		overdispersion_path,
		params_name
	)

	# Tumor, 10 samples
	params_name = "1cv_10.pickle"
	hyperparams["max_ppl"] = 10
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_10"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		overdispersion_path,
		params_name
	)

	# Tumor, low herit ASE
	list_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/tumor_fdr05.pickle"
	selection_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/tumor.pickle"

	hyperparams = {
		"prop_noise_eqtl": 0.95,
		"prop_noise_ase": 0.95,
		"min_causal": 1,
		"num_causal": 1.1,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"confidence": 0.95, 
		"model_flavors": "all"
	}

	# Tumor, all samples
	params_name = "1cv_all_low_herit.pickle"
	hyperparams["max_ppl"] = None
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all_low_herit"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		overdispersion_path,
		params_name
	)


	# # Kidney Data, multiple CV
	# input_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/jobs"
	# params_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/params"
	# script_path = os.path.join(curr_path, "job.py")

	# hyperparams = {
	# 	"prop_noise_eqtl": 0.95,
	# 	"prop_noise_ase": 0.50,
	# 	"min_causal": 1,
	# 	"num_causal": 1,
	# 	"search_mode": "shotgun",
	# 	"prob_threshold": 0.001,
	# 	"streak_threshold": 1000,
	# 	"search_iterations": 100000, 
	# 	"max_causal": 5,
	# 	"confidence": 0.95, 
	# 	"model_flavors": set(["full", "indep", "eqtl", "ase"])
	# }

	# num_tasks = 100
	# poll_freq = 5

	# # Normal
	# list_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/normal_fdr05.pickle"
	# selection_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/normal.pickle"

	# # Normal, all samples
	# params_name = "shotgun_all.pickle"
	# hyperparams["max_ppl"] = None
	# output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/shotgun_normal_all"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	params_path, 
	# 	hyperparams, 
	# 	num_tasks, 
	# 	poll_freq, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	overdispersion_path,
	# 	params_name
	# )

	# # Tumor
	# list_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/tumor_fdr05.pickle"
	# selection_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/tumor.pickle"

	# # Tumor, all samples
	# params_name = "shotgun_all.pickle"
	# hyperparams["max_ppl"] = None
	# output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/shotgun_tumor_all"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	params_path, 
	# 	hyperparams, 
	# 	num_tasks, 
	# 	poll_freq, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	overdispersion_path,
	# 	params_name
	# )

	# Prostate Data, 1 CV
	
	# Normal
	input_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_normal/jobs"
	params_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_normal/params"
	script_path = os.path.join(curr_path, "job.py")
	filter_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq/snp_filters/1KG_SNPs.pickle"
	overdispersion_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq/overdispersion/overdispersion.pickle"

	hyperparams = {
		"overdispersion": 0.05,
		"prop_noise_eqtl": 0.95,
		"prop_noise_ase": 0.50,
		"min_causal": 1,
		"num_causal": 1.1,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"confidence": 0.95, 
		"model_flavors": "all"
	}

	num_tasks = 100
	poll_freq = 5

	list_path = "all"
	selection_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq/sample_sets/normal.pickle"

	# Normal, all samples
	params_name = "1cv_all.pickle"
	hyperparams["max_ppl"] = None
	output_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_all"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		filter_path,
		overdispersion_path,
		params_name,
	)

	# Normal, 10 samples
	params_name = "1cv_10.pickle"
	hyperparams["max_ppl"] = 10
	output_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_10"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		filter_path,
		overdispersion_path,
		params_name
	)

	# Tumor
	input_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_tumor/jobs"
	params_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_tumor/params"
	script_path = os.path.join(curr_path, "job.py")

	hyperparams = {
		"overdispersion": 0.05,
		"prop_noise_eqtl": 0.95,
		"prop_noise_ase": 0.50,
		"min_causal": 1,
		"num_causal": 1.1,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"confidence": 0.95, 
		"model_flavors": "all"
	}

	num_tasks = 100
	poll_freq = 5

	list_path = "all"
	selection_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq/sample_sets/tumor.pickle"

	# Tumor, all samples
	params_name = "1cv_all.pickle"
	hyperparams["max_ppl"] = None
	output_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_all"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		filter_path,
		overdispersion_path,
		params_name
	)

	# Normal, 10 samples
	params_name = "1cv_10.pickle"
	hyperparams["max_ppl"] = 10
	output_path = "/bcb/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_10"

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path,
		selection_path,
		list_path,
		filter_path,
		overdispersion_path,
		params_name
	)