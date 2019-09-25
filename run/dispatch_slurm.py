import numpy as np
import os
import time
import pickle
import subprocess
import shutil

def dispatch(
		targets, 
		batch_num,
		job_data_path,
		output_path, 
		input_path, 
		params_path,
		script_path, 
		selection_path, 
		filter_path, 
		overdispersion_path
	):
	io_data = []

	for t in targets:
		if t is not None:
			job_input_path = os.path.join(input_path, t, "input.pickle")
			job_output_path = os.path.join(output_path, t)
			data_tuple = (job_input_path, job_output_path)
			io_data.append(data_tuple)

			if not os.path.exists(job_output_path):
				os.makedirs(job_output_path)

	if not os.path.exists(job_data_path):
		os.makedirs(job_data_path)

	io_name =  os.path.join(job_data_path, "{0}_io.pickle".format(batch_num))
	with open(io_name, "wb") as io_file:
		pickle.dump(io_data, io_file)

	out_name = os.path.join(job_data_path, "{0}_stdout.txt".format(batch_num))
	err_name = os.path.join(job_data_path, "{0}_stderr.txt".format(batch_num))
	job_args = [
		"sbatch", 
		"-J", 
		str(batch_num), 
		"-o",
		out_name,
		"-e",
		err_name,
		"-x",
		"node05,node15",
		"--mem",
		"3500",
		script_path	
	]
	job_args.extend([io_name, params_path, selection_path, filter_path, overdispersion_path])

	timeout = "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"
	# print(" ".join(job_args)) ####
	# raise Exception ####
	while True:
		try:
			submission = subprocess.run(job_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			print(str(submission.stdout, 'utf-8').rstrip())
			break
		except subprocess.CalledProcessError as e:
			# print(e.stdout) ####
			err = str(e.stderr, 'utf-8').rstrip()
			print(err)
			if err == timeout:
				print("Retrying Submit")
				continue
			else:
				raise e

def run(
		output_path, 
		input_path, 
		job_data_path,
		params_path, 
		hyperparams,  
		script_path, 
		selection_path, 
		list_path, 
		filter_path, 
		overdispersion_path,
		params_name,
		batch_size
	):
	shutil.rmtree(output_path, ignore_errors=True)
	if not os.path.exists(output_path):
		os.makedirs(output_path)

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

	# targets = ["ENSG00000134575.5"] ####

	num_targets = len(targets)
	num_padding = (-num_targets) % batch_size
	num_jobs = (num_targets + num_padding) / batch_size
	# print(num_jobs) ####
	targets.extend([None] * num_padding)
	batches = np.reshape(targets, (int(num_jobs), batch_size))

	for i, b in enumerate(batches):
		dispatch(
			b, 
			i,
			job_data_path,
			output_path, 
			input_path, 
			hyperparams_path, 
			script_path, 
			selection_path,
			filter_path,
			overdispersion_path
		)

if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))
	batch_size = 15

	# # Kidney Data, 1 CV
	# input_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/jobs"
	# params_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/params"
	# script_path = os.path.join(curr_path, "job.py")
	# # filter_path = "all"
	# filter_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/snp_filters/1KG_SNPs.pickle"
	# overdispersion_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/overdispersion/overdispersion.pickle"

	# hyperparams = {
	# 	"total_exp_herit_prior": 0.05,
	# 	"imbalance_herit_prior": 0.40,
	# 	"cross_corr_prior": 0.9,
	# 	"min_causal": 1,
	# 	"num_causal": 1.,
	# 	"search_mode": "exhaustive",
	# 	"max_causal": 1,
	# 	"confidence": 0.95, 
	# 	"model_flavors": "all"
	# }

	# # Normal
	# list_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/normal_fdr05.pickle"
	# selection_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/normal.pickle"

	# # Normal, all samples
	# params_name = "1cv_all.pickle"
	# hyperparams["max_ppl"] = None
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_all"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/1cv_normal_all"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams,  
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Normal, 50 samples
	# params_name = "1cv_50.pickle"
	# hyperparams["max_ppl"] = 50
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_50"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/1cv_normal_50"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path, 
	# 	params_path, 
	# 	hyperparams,  
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Normal, 10 samples
	# params_name = "1cv_10.pickle"
	# hyperparams["max_ppl"] = 10
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_10"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/1cv_normal_10"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path, 
	# 	params_path, 
	# 	hyperparams, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Tumor
	# list_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/tumor_fdr05.pickle"
	# # list_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/verified.pickle" ####
	# selection_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/tumor.pickle"

	# # Tumor, all samples
	# params_name = "1cv_all.pickle"
	# hyperparams["max_ppl"] = None
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/1cv_tumor_all"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Tumor, 200 samples
	# params_name = "1cv_200.pickle"
	# hyperparams["max_ppl"] = 200
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_200"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/1cv_tumor_200"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Tumor, 100 samples
	# params_name = "1cv_100.pickle"
	# hyperparams["max_ppl"] = 100
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_100"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/1cv_tumor_100"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams,  
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Tumor, 50 samples
	# params_name = "1cv_50.pickle"
	# hyperparams["max_ppl"] = 50
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_50"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/1cv_tumor_50"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Tumor, 10 samples
	# params_name = "1cv_10.pickle"
	# hyperparams["max_ppl"] = 10
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_10"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/1cv_tumor_10"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams,  
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Tumor, low herit ASE
	# list_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/tumor_fdr05.pickle"
	# selection_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/tumor.pickle"

	# hyperparams = {
	# 	"total_exp_herit_prior": 0.05,
	# 	"imbalance_herit_prior": 0.05,
	# 	"cross_corr_prior": 0.9,
	# 	"min_causal": 1,
	# 	"num_causal": 1.1,
	# 	"search_mode": "exhaustive",
	# 	"max_causal": 1,
	# 	"confidence": 0.95, 
	# 	"model_flavors": "all"
	# }

	# # Tumor, all samples
	# params_name = "1cv_all_low_herit.pickle"
	# hyperparams["max_ppl"] = None
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all_low_herit"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/1cv_tumor_all_low_herit"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )


	# Kidney Data, multiple CV
	input_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/jobs"
	params_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/params"
	script_path = os.path.join(curr_path, "job.py")
	filter_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/snp_filters/1KG_SNPs.pickle"
	overdispersion_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/overdispersion/overdispersion.pickle"

	hyperparams = {
		"total_exp_herit_prior": 0.01,
		"imbalance_herit_prior": 0.1,
		"cross_corr_prior": 0.9,
		"min_causal": 1,
		"max_causal": 3,
		"num_causal": 1,
		"search_mode": "shotgun",
		"prob_threshold": 0.001,
		"streak_threshold": 1000,
		"search_iterations": 100000, 
		"confidence": 0.95, 
		"model_flavors": set(["full", "indep", "eqtl", "ase", "fmb"]),
	}
	# hyperparams["model_flavors"] = set(["fmb"]) ####

	# # Normal
	# list_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/normal_fdr05.pickle"
	# # list_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/shotgun_normal_fail.pickle"
	# selection_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/normal.pickle"

	# # Normal, all samples
	# params_name = "shotgun_all.pickle"
	# hyperparams["max_ppl"] = None
	# output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/shotgun_normal_all"
	# job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/shotgun_normal_all"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams,  
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# Tumor
	list_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/tumor_fdr05.pickle"
	# list_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/shotgun_tumor_fail.pickle"
	selection_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/sample_sets/tumor.pickle"

	# Tumor, all samples
	params_name = "shotgun_all.pickle"
	hyperparams["max_ppl"] = None
	output_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/shotgun_tumor_all"
	job_data_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/job_data/shotgun_tumor_all"

	run(
		output_path, 
		input_path, 
		job_data_path,
		params_path, 
		hyperparams, 
		script_path,
		selection_path,
		list_path,
		filter_path,
		overdispersion_path,
		params_name,
		batch_size
	)

	# # Prostate Data, 1 CV
	
	# # Normal
	# input_path = "/agusevlab/awang/job_data/prostate_chipseq_normal/jobs"
	# params_path = "/agusevlab/awang/job_data/prostate_chipseq_normal/params"
	# script_path = os.path.join(curr_path, "job.py")
	# filter_path = "/agusevlab/awang/job_data/prostate_chipseq/snp_filters/1KG_SNPs.pickle"
	# overdispersion_path = "/agusevlab/awang/job_data/prostate_chipseq/overdispersion/overdispersion.pickle"

	# hyperparams = {
	# 	"overdispersion": 4.22e-4,
	# 	"prop_noise_eqtl": 0.95,
	# 	"prop_noise_ase": 0.50,
	# 	"min_causal": 1,
	# 	"num_causal": 1.1,
	# 	"search_mode": "exhaustive",
	# 	"max_causal": 1,
	# 	"confidence": 0.95, 
	# 	"model_flavors": "all"
	# }

	# list_path = "all"
	# selection_path = "/agusevlab/awang/job_data/prostate_chipseq/sample_sets/normal.pickle"

	# # Normal, all samples
	# params_name = "1cv_all.pickle"
	# hyperparams["max_ppl"] = None
	# output_path = "/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_all"
	# job_data_path = "/agusevlab/awang/job_data/prostate_chipseq_normal/job_data/1cv_normal_all"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Normal, 10 samples
	# params_name = "1cv_10.pickle"
	# hyperparams["max_ppl"] = 10
	# output_path = "/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_10"
	# job_data_path = "/agusevlab/awang/job_data/prostate_chipseq_normal/job_data/1cv_normal_10"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Tumor
	# input_path = "/agusevlab/awang/job_data/prostate_chipseq_tumor/jobs"
	# params_path = "/agusevlab/awang/job_data/prostate_chipseq_tumor/params"
	# script_path = os.path.join(curr_path, "job.py")
	# filter_path = "/agusevlab/awang/job_data/prostate_chipseq/snp_filters/1KG_SNPs.pickle"
	# overdispersion_path = "/agusevlab/awang/job_data/prostate_chipseq/overdispersion/overdispersion.pickle"

	# hyperparams = {
	# 	"total_exp_herit_prior": 0.05,
	# 	"imbalance_herit_prior": 0.40,
	# 	"cross_corr_prior": 0.9,
	# 	"min_causal": 1,
	# 	"num_causal": 1.1,
	# 	"search_mode": "exhaustive",
	# 	"max_causal": 1,
	# 	"confidence": 0.95, 
	# 	"model_flavors": "all"
	# }

	# list_path = "all"
	# selection_path = "/agusevlab/awang/job_data/prostate_chipseq/sample_sets/tumor.pickle"

	# # Tumor, all samples
	# params_name = "1cv_all.pickle"
	# hyperparams["max_ppl"] = None
	# output_path = "/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_all"
	# job_data_path = "/agusevlab/awang/job_data/prostate_chipseq_normal/job_data/1cv_tumor_all"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams,  
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )

	# # Tumor, 10 samples
	# params_name = "1cv_10.pickle"
	# hyperparams["max_ppl"] = 10
	# output_path = "/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_10"
	# job_data_path = "/agusevlab/awang/job_data/prostate_chipseq_normal/job_data/1cv_tumor_10"

	# run(
	# 	output_path, 
	# 	input_path, 
	# 	job_data_path,
	# 	params_path, 
	# 	hyperparams, 
	# 	script_path,
	# 	selection_path,
	# 	list_path,
	# 	filter_path,
	# 	overdispersion_path,
	# 	params_name,
	# 	batch_size
	# )