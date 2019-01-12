from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

try:
	import cPickle as pickle
except ImportError:
	import pickle

def write_thresholds(summary, out_dir, total_jobs):
	thresholds_list = []
	if "full" in model_flavors:
		thresholds_list.append("Full Model")
		for k in sorted(summary["thresholds_full"].keys()):
			thresholds_list.append("{0}\t{1}".format(k, summary["thresholds_full"][k] / total_jobs))
		thresholds_list.append("")
	if "indep" in model_flavors:
		thresholds_list.append("Independent Likelihoods")
		for k in sorted(summary["thresholds_indep"].keys()):
			thresholds_list.append("{0}\t{1}".format(k, summary["thresholds_indep"][k] / total_jobs))
		thresholds_list.append("")
	if "eqtl" in model_flavors:
		thresholds_list.append("eQTL-Only")
		for k in sorted(summary["thresholds_eqtl"].keys()):
			thresholds_list.append("{0}\t{1}".format(k, summary["thresholds_eqtl"][k] / total_jobs))
		thresholds_list.append("")
	if "ase" in model_flavors:
		thresholds_list.append("ASE-Only")
		for k in sorted(summary["thresholds_ase"].keys()):
			thresholds_list.append("{0}\t{1}".format(k, summary["thresholds_ase"][k] / total_jobs))
		thresholds_list.append("")
	if "acav" in model_flavors:
		thresholds_list.append("CAVIAR-ASE")
		for k in sorted(summary["thresholds_caviar_ase"].keys()):
			thresholds_list.append("{0}\t{1}".format(k, summary["thresholds_caviar_ase"][k] / total_jobs))
		thresholds_list.append("")

	thresholds_str = "\n".join(thresholds_list)
	out_path = os.path.join(out_dir, "causal_set_thresholds")

	with open(out_path, "w") as out_file:
		out_file.write(thresholds_str)

def plot_dist(result, out_dir, name):
	sns.set(style="white")

	if "full" in model_flavors:
		set_sizes_full = result["set_sizes_full"]
		try:
			sns.distplot(
				set_sizes_full,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 3, "shade":True},
				label="Full"
			)
		except Exception:
			pass
	if "indep" in model_flavors:
		set_sizes_indep = result["set_sizes_indep"]
		try:
			sns.distplot(
				set_sizes_indep,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 3, "shade":True},
				label="Independent Likelihoods"
			)
		except Exception:
			pass
	if "eqtl" in model_flavors:
		set_sizes_eqtl = result["set_sizes_eqtl"]
		try:
			sns.distplot(
				set_sizes_eqtl,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 3, "shade":True},
				label="eQTL-Only"
			)
		except Exception:
			pass
	if "ase" in model_flavors:
		set_sizes_ase = result["set_sizes_ase"]
		try:
			sns.distplot(
				set_sizes_ase,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 3, "shade":True},
				label="ASE-Only"
			)
		except Exception:
			pass
	if "acav" in model_flavors:
		set_sizes_caviar_ase = result["set_sizes_caviar_ase"]
		try:
			sns.distplot(
				set_sizes_caviar_ase,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 3, "shade":True},
				label="CAVIAR-ASE"
			)
		except Exception:
			pass
	plt.xlim(0, None)
	plt.legend(title="Model")
	plt.xlabel("Set Size")
	plt.ylabel("Density")
	plt.title("Distribution of Causal Set Sizes: {0}".format(name))
	plt.savefig(os.path.join(out_dir, "set_size_distribution.svg"))
	plt.clf()

def plot_series(series, primary_var_vals, primary_var_name, out_dir, name, model_flavors):
	dflst = []
	for key, val in series.viewitems():
		if "full" in model_flavors:
			for skey, sval in series["all_sizes_full"].viewitems():
				for i in sval:
					dflst.append([i, skey, "Full"])
		if "indep" in model_flavors:
			for skey, sval in series["all_sizes_indep"].viewitems():
				for i in sval:
					dflst.append([i, skey, "Independent Likelihoods"])
		if "eqtl" in model_flavors:
			for skey, sval in series["all_sizes_eqtl"].viewitems():
				for i in sval:
					dflst.append([i, skey, "eQTL-Only"])
		if "ase" in model_flavors:
			for skey, sval in series["all_sizes_ase"].viewitems():
				for i in sval:
					dflst.append([i, skey, "ASE-Only"])
		if "acav" in model_flavors:
			for skey, sval in series["all_sizes_caviar_ase"].viewitems():
				for i in sval:
					dflst.append([i, skey, "CAVIAR-ASE"])
	res_df = pd.DataFrame(dflst, columns=["Set Size", primary_var_name, "Model"])

	title = "Causal Set Sizes Across {0}:\n{1}".format(primary_var_name, name)
	
	sns.set(style="whitegrid")
	sns.violinplot(
		x=primary_var_name,
		y="Set Size",
		hue="Model",
		data=res_df,
		order=primary_var_vals
	)
	plt.title(title)
	plt.savefig(os.path.join(out_dir, "causal_sets_violin.svg"))
	plt.clf()

	sns.barplot(
		x=primary_var_name, 
		y="Set Size",
		hue="Model",
		data=res_df,
		order=primary_var_vals
	)
	plt.title(title)
	plt.savefig(os.path.join(out_dir, "causal_sets_bar.svg"))
	plt.clf()

def interpret(target_dir, out_dir, name, model_flavors):
	if model_flavors == "all":
		model_flavors = set(["full", "indep", "eqtl", "ase", "acav"])

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	targets = os.listdir(target_dir)

	summary = {}
	if "full" in model_flavors:
		summary["set_sizes_full"] = []
		summary["thresholds_full"] = {5: 0, 10: 0, 15: 0, 20: 0}
	if "indep" in model_flavors:
		summary["set_sizes_indep"] = []
		summary["thresholds_indep"] = {5: 0, 10: 0, 15: 0, 20: 0}
	if "eqtl" in model_flavors:
		summary["set_sizes_eqtl"] = []
		summary["thresholds_eqtl"] = {5: 0, 10: 0, 15: 0, 20: 0}
	if "ase" in model_flavors:
		summary["set_sizes_ase"] = []
		summary["thresholds_ase"] = {5: 0, 10: 0, 15: 0, 20: 0}
	if "acav" in model_flavors:
		summary["set_sizes_caviar_ase"] = []
		summary["thresholds_caviar_ase"] = {5: 0, 10: 0, 15: 0, 20: 0}

	failed_jobs = []
	successes = 0

	for t in targets:
		# print(t) ####
		result_path = os.path.join(target_dir, t, "output.pickle")

		try:
			with open(result_path, "rb") as result_file:
				result = pickle.load(result_file)
		except (EOFError, IOError):
			failed_jobs.append(t)
			continue

		if "full" in model_flavors:
			set_size = np.count_nonzero(result["causal_set_full"])
			summary["set_sizes_full"].append(set_size)
			for k, v in summary["thresholds_full"].viewitems():
				if set_size <= k:
					v += 1
		if "indep" in model_flavors:
			set_size = np.count_nonzero(result["causal_set_indep"])
			summary["set_sizes_indep"].append(set_size)
			for k, v in summary["thresholds_indep"].viewitems():
				if set_size <= k:
					v += 1
		if "eqtl" in model_flavors:
			set_size = np.count_nonzero(result["causal_set_eqtl"])
			summary["set_sizes_eqtl"].append(set_size)
			for k, v in summary["thresholds_eqtl"].viewitems():
				if set_size <= k:
					v += 1
		if "ase" in model_flavors:
			set_size = np.count_nonzero(result["causal_set_ase"])
			summary["set_sizes_ase"].append(set_size)
			for k, v in summary["thresholds_ase"].viewitems():
				if set_size <= k:
					v += 1
		if "acav" in model_flavors:
			set_size = np.count_nonzero(result["causal_set_caviar_ase"])
			summary["set_sizes_caviar_ase"].append(set_size)
			for k, v in summary["thresholds_caviar_ase"].viewitems():
				if set_size <= k:
					v += 1

		successes += 1

	with open(os.path.join(out_dir, "failed_jobs.txt"), "w") as fail_out:
		fail_out.write("\n".join(failed_jobs))
	
	write_thresholds(summary, out_dir, primary_var_vals, primary_var_names, successes)
	plot_dist(summary, out_dir, name)

	return summary

def interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name):
	if model_flavors == "all":
		model_flavors = set(["full", "indep", "eqtl", "ase", "acav"])

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	series = {}
	if "full" in model_flavors:
		series["avg_sets_full"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_full"] = {}
		series["counts_full"] = {i: 0 for i in primary_var_vals}
	if "indep" in model_flavors:
		series["avg_sets_indep"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_indep"] = {}
		series["counts_indep"] = {i: 0 for i in primary_var_vals}
	if "eqtl" in model_flavors:
		series["avg_sets_eqtl"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_eqtl"] = {}
		series["counts_eqtl"] = {i: 0 for i in primary_var_vals}
	if "ase" in model_flavors:
		series["avg_sets_ase"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_ase"] = {}
		series["counts_ase"] = {i: 0 for i in primary_var_vals}
	if "acav" in model_flavors:
		series["avg_sets_caviar_ase"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_caviar_ase"] = {}
		series["counts_caviar_ase"] = {i: 0 for i in primary_var_vals}

	for ind, val in enumerate(summaries):
		var_val = primary_var_vals[ind]
		if "full" in model_flavors:
			series["avg_sets_full"][var_val] = np.mean(val["set_sizes_full"])
			series["all_sizes_full"][var_val] = val["set_sizes_full"]
			series["counts_full"][var_val] = len(val["set_sizes_full"])
		if "indep" in model_flavors:
			series["avg_sets_indep"][var_val] = np.mean(val["set_sizes_indep"])
			series["all_sizes_indep"][var_val] = val["set_sizes_indep"]
			series["counts_indep"][var_val] = len(val["set_sizes_indep"])
		if "eqtl" in model_flavors:
			series["avg_sets_eqtl"][var_val] = np.mean(val["set_sizes_eqtl"])
			series["all_sizes_eqtl"][var_val] = val["set_sizes_eqtl"]
			series["counts_eqtl"][var_val] = len(val["set_sizes_eqtl"])
		if "ase" in model_flavors:
			series["avg_sets_ase"][var_val] = np.mean(val["set_sizes_ase"])
			series["all_sizes_ase"][var_val] = val["set_sizes_ase"]
			series["counts_ase"][var_val] = len(val["set_sizes_ase"])
		if "acav" in model_flavors:
			series["avg_sets_caviar_ase"][var_val] = np.mean(val["set_sizes_caviar_ase"])
			series["all_sizes_caviar_ase"][var_val] = val["set_sizes_caviar_ase"]
			series["counts_caviar_ase"][var_val] = len(val["set_sizes_caviar_ase"])

	plot_series(series, primary_var_vals, primary_var_name, out_dir, name, model_flavors)

if __name__ == '__main__':
	# Normal

	# Normal, all samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_all"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_all"
	name = "Kidney RNA-Seq\nAll Normal Samples"

	normal_all = interpret(target_dir, out_dir, name)

	# Normal, 50 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_50"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_50"
	name = "Kidney RNA-Seq\n50 Normal Samples"

	normal_50 = interpret(target_dir, out_dir, name)

	# Normal, 10 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_10"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_10"
	name = "Kidney RNA-Seq\n10 Normal Samples"

	normal_10 = interpret(target_dir, out_dir, name)

	# Normal, across sample sizes
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_sample_sizes"
	name = "Kidney RNA-Seq"
	model_flavors = set(["full", "eqtl", "acav"])
	summaries = [normal_all, normal_50, normal_10]
	primary_var_vals = [90, 50, 10]
	primary_var_name = "Sample Size"

	interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name)

	# Tumor

	# Tumor, all samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_all"
	name = "Kidney RNA-Seq\nAll Tumor Samples"

	interpret(target_dir, out_dir, name)

	# Tumor, 200 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_200"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_200"
	name = "Kidney RNA-Seq\n200 Tumor Samples"

	interpret(target_dir, out_dir, name)

	# Tumor, 100 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_100"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_100"
	name = "Kidney RNA-Seq\n100 Tumor Samples"

	interpret(target_dir, out_dir, name)

	# Tumor, 50 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_50"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_50"
	name = "Kidney RNA-Seq\n50 Tumor Samples"

	interpret(target_dir, out_dir, name)

	# Tumor, 10 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_10"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_10"
	name = "Kidney RNA-Seq\n10 Tumor Samples"

	interpret(target_dir, out_dir, name)