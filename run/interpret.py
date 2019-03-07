from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

try:
	import cPickle as pickle
except ImportError:
	import pickle

def write_thresholds(summary, out_dir, total_jobs, model_flavors):
	thresholds_list = []
	if "full" in model_flavors:
		thresholds_list.append("Joint-Correlated")
		for k in sorted(summary["thresholds_full"].keys()):
			thresholds_list.append("{0}\t{1}".format(k, summary["thresholds_full"][k] / total_jobs))
		thresholds_list.append("")
	if "indep" in model_flavors:
		thresholds_list.append("Joint-Independent")
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

def write_size_probs(summary, out_dir, total_jobs, model_flavors):
	size_probs_list = []
	if "full" in model_flavors:
		size_probs_list.append("Joint-Correlated")
		size_probs_list.append("\t".join(str(i) for i in summary["size_probs_full"] / total_jobs))
		size_probs_list.append("")
	if "indep" in model_flavors:
		size_probs_list.append("Joint-Independent")
		size_probs_list.append("\t".join(str(i) for i in summary["size_probs_indep"] / total_jobs))
		size_probs_list.append("")
	if "eqtl" in model_flavors:
		size_probs_list.append("eQTL-Only")
		size_probs_list.append("\t".join(str(i) for i in summary["size_probs_eqtl"] / total_jobs))
		size_probs_list.append("")
	if "ase" in model_flavors:
		size_probs_list.append("ASE-Only")
		size_probs_list.append("\t".join(str(i) for i in summary["size_probs_ase"] / total_jobs))
		size_probs_list.append("")

	size_probs_str = "\n".join(size_probs_list)
	out_path = os.path.join(out_dir, "causal_set_size_probabilities")

	with open(out_path, "w") as out_file:
		out_file.write(size_probs_str)

def plot_dist(result, out_dir, name, model_flavors, metric, cumu):
	if metric == "size":
		kwd = "set_sizes"
	elif metric == "prop":
		kwd = "set_props"

	sns.set(style="whitegrid", font="Roboto")

	# if "full" in model_flavors:
	# 	set_sizes_full = result["{0}_full".format(kwd)]
	# 	sns.distplot(
	# 		set_sizes_full,
	# 		hist=False,
	# 		kde=True,
	# 		kde_kws={"linewidth": 2, "shade":True, "cumulative":cumu},
	# 		label="Full"
	# 	)

	if "full" in model_flavors:
		set_sizes_full = result["{0}_full".format(kwd)]
		try:
			sns.distplot(
				set_sizes_full,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 2, "shade":False, "cumulative":cumu},
				label="Joint-Correlated"
			)
		except Exception:
			pass
	if "indep" in model_flavors:
		set_sizes_indep = result["{0}_indep".format(kwd)]
		try:
			sns.distplot(
				set_sizes_indep,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 2, "shade":False, "cumulative":cumu},
				label="Joint-Independent"			
			)
		except Exception:
			pass
	if "eqtl" in model_flavors:
		set_sizes_eqtl = result["{0}_eqtl".format(kwd)]
		try:
			sns.distplot(
				set_sizes_eqtl,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 2, "shade":False, "cumulative":cumu},
				label="eQTL-Only"			
			)
		except Exception:
			pass
	if "ase" in model_flavors:
		set_sizes_ase = result["{0}_ase".format(kwd)]
		try:
			sns.distplot(
				set_sizes_ase,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 2, "shade":False, "cumulative":cumu},
				label="ASE-Only"			
			)
		except Exception:
			pass
	if "acav" in model_flavors:
		set_sizes_caviar_ase = result["{0}_caviar_ase".format(kwd)]
		try:
			sns.distplot(
				set_sizes_caviar_ase,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 2, "shade":False, "cumulative":cumu},
				label="CAVIAR-ASE"			
			)
		except Exception:
			pass
	if metric == "prop":
		plt.xlim(0, 1)
	elif metric == "size":
		plt.xlim(0, 1000)
	plt.legend(title="Model")
	if cumu:
		cumu_kwd = "Cumulative "
		cumu_fname = "_cumu"
	else:
		cumu_kwd = ""
		cumu_fname = ""
	if metric == "size":
		plt.xlabel("Set Size")
		plt.ylabel("Density")
		plt.title("{0}Distribution of Causal Set Sizes: {1}".format(cumu_kwd, name))
		plt.savefig(os.path.join(out_dir, "set_size_distribution{0}.svg".format(cumu_fname)))
	elif metric == "prop":
		plt.xlabel("Set Size (Proportion of Total Markers)")
		plt.ylabel("Density")
		plt.title("{0}Distribution of Causal Set Sizes: {1}".format(cumu_kwd, name))
		plt.savefig(os.path.join(out_dir, "set_prop_distribution{0}.svg".format(cumu_fname)))
	plt.clf()

def plot_series(series, primary_var_vals, primary_var_name, out_dir, name, model_flavors, metric):
	if metric == "size":
		kwd = "all_sizes"
		label = "Set Size"
		filename = "causal_set_sizes"
	elif metric == "prop":
		kwd = "all_props"
		label = "Set Size (Proportion of Markers)"
		filename = "causal_set_props"

	dflst = []
	for key, val in series.viewitems():
		if "full" in model_flavors:
			for skey, sval in series["{0}_full".format(kwd)].viewitems():
				for i in sval:
					dflst.append([i, skey, "Joint-Correlated"])
		if "indep" in model_flavors:
			for skey, sval in series["{0}_indep".format(kwd)].viewitems():
				for i in sval:
					dflst.append([i, skey, "Joint-Independent"])
		if "eqtl" in model_flavors:
			for skey, sval in series["{0}_eqtl".format(kwd)].viewitems():
				for i in sval:
					dflst.append([i, skey, "eQTL-Only"])
		if "ase" in model_flavors:
			for skey, sval in series["{0}_ase".format(kwd)].viewitems():
				for i in sval:
					dflst.append([i, skey, "ASE-Only"])
		if "acav" in model_flavors:
			for skey, sval in series["{0}_caviar_ase".format(kwd)].viewitems():
				for i in sval:
					dflst.append([i, skey, "CAVIAR-ASE"])
	res_df = pd.DataFrame(dflst, columns=[label, primary_var_name, "Model"])

	title = "Causal Set Sizes Across {0}:\n{1}".format(primary_var_name, name)
	
	sns.set(style="whitegrid", font="Roboto")
	if metric == "prop":
		plt.ylim(0, 1)
	elif metric == "size":
		plt.ylim(0, 1000)
	sns.violinplot(
		x=primary_var_name,
		y=label,
		hue="Model",
		data=res_df,
		order=primary_var_vals
	)
	plt.title(title)
	plt.savefig(os.path.join(out_dir, "{0}_violin.svg".format(filename)))
	plt.clf()

	sns.barplot(
		x=primary_var_name, 
		y=label,
		hue="Model",
		data=res_df,
		order=primary_var_vals
	)
	plt.title(title)
	plt.savefig(os.path.join(out_dir, "{0}_bar.svg".format(filename)))
	plt.clf()

def plot_recall(series, primary_var_vals, primary_var_name, out_dir, name, model_flavors):
	print(model_flavors) ####
	print(series.keys()) ####
	dflst = []
	for key, val in series.viewitems():
		if "full" in model_flavors:
			for skey, sval in series["recall_full"].viewitems():
				data = sorted(sval.items(), key=lambda x: x[0])
				cumu_recall = 0.
				for x, val in data:
					cumu_recall += val
					dflst.append([x, cumu_recall, skey, "Joint-Correlated"])

		if "indep" in model_flavors:
			for skey, sval in series["recall_indep"].viewitems():
				data = sorted(sval.items(), key=lambda x: x[0])
				cumu_recall = 0.
				for x, val in data:
					cumu_recall += val
					dflst.append([x, cumu_recall, skey, "Joint-Independent"])
		if "eqtl" in model_flavors:
			for skey, sval in series["recall_eqtl"].viewitems():
				data = sorted(sval.items(), key=lambda x: x[0])
				cumu_recall = 0.
				for x, val in data:
					cumu_recall += val
					dflst.append([x, cumu_recall, skey, "eQTL-Only"])
		if "ase" in model_flavors:
			for skey, sval in series["recall_ase"].viewitems():
				data = sorted(sval.items(), key=lambda x: x[0])
				cumu_recall = 0.
				for x, val in data:
					cumu_recall += val
					dflst.append([x, cumu_recall, skey, "ASE-Only"])
		if "acav" in model_flavors:
			for skey, sval in series["recall_caviar_ase"].viewitems():
				data = sorted(sval.items(), key=lambda x: x[0])
				cumu_recall = 0.
				for x, val in data:
					cumu_recall += val
					dflst.append([x, cumu_recall, skey, "CAVIAR-ASE"])
	res_df = pd.DataFrame(dflst, columns=["Proportion of Selected Markers", "Inclusion Rate", primary_var_name, "Model"])
	# print(res_df) ####
	
	title = "Inclusion Rates Across {0}:\n{1}".format(primary_var_name, name)
	
	sns.set(style="whitegrid", font="Roboto")
	palette = sns.cubehelix_palette(len(primary_var_vals))
	sns.lineplot(
		x="Proportion of Selected Markers",
		y="Inclusion Rate",
		hue=primary_var_name,
		style="Model",
		data=res_df,
		sort=True,
		palette=palette
	)
	plt.title(title)
	plt.savefig(os.path.join(out_dir, "inclusion.svg"))
	plt.clf()

	if "full" in model_flavors:
		title = "Proportion Rates Across {0}:\n{1}, {2} Model".format(primary_var_name, name, "Joint-Correlated")
		sns.set(style="whitegrid", font="Roboto")
		palette = sns.cubehelix_palette(len(primary_var_vals))
		sns.lineplot(
			x="Proportion of Selected Markers",
			y="Inclusion Rate",
			hue=primary_var_name,
			data=res_df.query("Model == 'Joint-Correlated'"),
			sort=True,
			palette=palette
		)
		plt.title(title)
		plt.savefig(os.path.join(out_dir, "inclusion_full.svg"))
		plt.clf()
	if "indep" in model_flavors:
		title = "Inclusion Rates Across {0}:\n{1}, {2} Model".format(primary_var_name, name, "Joint-Independent")
		sns.set(style="whitegrid", font="Roboto")
		palette = sns.cubehelix_palette(len(primary_var_vals))
		sns.lineplot(
			x="Proportion of Selected Markers",
			y="Inclusion Rate",
			hue=primary_var_name,
			data=res_df.query("Model == 'Joint-Independent'"),
			sort=True,
			palette=palette
		)
		plt.title(title)
		plt.savefig(os.path.join(out_dir, "inclusion_indep.svg"))
		plt.clf()
	if "eqtl" in model_flavors:
		title = "Inclusion Rates Across {0}:\n{1}, {2} Model".format(primary_var_name, name, "eQTL-Only")
		sns.set(style="whitegrid", font="Roboto")
		palette = sns.cubehelix_palette(len(primary_var_vals))
		sns.lineplot(
			x="Proportion of Selected Markers",
			y="Inclusion Rate",
			hue=primary_var_name,
			data=res_df.query("Model == 'eQTL-Only'"),
			sort=True,
			palette=palette
		)
		plt.title(title)
		plt.savefig(os.path.join(out_dir, "inclusion_eqtl.svg"))
		plt.clf()
	if "ase" in model_flavors:
		title = "Inclusion Rates Across {0}:\n{1}, {2} Model".format(primary_var_name, name, "ASE-Only")
		sns.set(style="whitegrid", font="Roboto")
		palette = sns.cubehelix_palette(len(primary_var_vals))
		sns.lineplot(
			x="Proportion of Selected Markers",
			y="Inclusion Rate",
			hue=primary_var_name,
			data=res_df.query("Model == 'ASE-Only'"),
			sort=True,
			palette=palette
		)
		plt.title(title)
		plt.savefig(os.path.join(out_dir, "inclusion_ase.svg"))
		plt.clf()
	if "acav" in model_flavors:
		title = "Inclusion Rates Across {0}:\n{1}, {2} Model".format(primary_var_name, name, "CAVIAR-ASE")
		sns.set(style="whitegrid", font="Roboto")
		palette = sns.cubehelix_palette(len(primary_var_vals))
		sns.lineplot(
			x="Proportion of Selected Markers",
			y="Inclusion Rate",
			hue=primary_var_name,
			data=res_df.query("Model == 'CAVIAR-ASE'"),
			sort=True,
			palette=palette
		)
		plt.title(title)
		plt.savefig(os.path.join(out_dir, "inclusion_acav.svg"))
		plt.clf()

def interpret(target_dir, out_dir, name, model_flavors):
	if model_flavors == "all":
		model_flavors = set(["full", "indep", "eqtl", "ase", "acav"])

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	targets = os.listdir(target_dir)

	summary = {"names":[]}
	if "full" in model_flavors:
		summary["causal_sets_full"] = []
		summary["ppas_full"] = []
		summary["set_sizes_full"] = []
		summary["set_props_full"] = []
		summary["thresholds_full"] = {5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
		summary["size_probs_full"] = np.zeros(6)
	if "indep" in model_flavors:
		summary["causal_sets_indep"] = []
		summary["ppas_indep"] = []
		summary["set_sizes_indep"] = []
		summary["set_props_indep"] = []
		summary["thresholds_indep"] = {5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
		summary["size_probs_indep"] = np.zeros(6)
	if "eqtl" in model_flavors:
		summary["causal_sets_eqtl"] = []
		summary["ppas_eqtl"] = []
		summary["set_sizes_eqtl"] = []
		summary["set_props_eqtl"] = []
		summary["thresholds_eqtl"] = {5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
		summary["size_probs_eqtl"] = np.zeros(6)
	if "ase" in model_flavors:
		summary["causal_sets_ase"] = []
		summary["ppas_ase"] = []
		summary["set_sizes_ase"] = []
		summary["set_props_ase"] = []
		summary["thresholds_ase"] = {5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
		summary["size_probs_ase"] = np.zeros(6)
	if "acav" in model_flavors:
		summary["causal_sets_caviar_ase"] = []
		summary["ppas_caviar_ase"] = []
		summary["set_sizes_caviar_ase"] = []
		summary["set_props_caviar_ase"] = []
		summary["thresholds_caviar_ase"] = {5: 0, 10: 0, 20: 0, 50: 0, 100: 0}

	failed_jobs = []
	insufficient_data_jobs = []
	insufficient_snps_jobs = []
	successes = 0

	for t in targets:
		# print(t) ####
		result_path = os.path.join(target_dir, t, "output.pickle")
		stdout_path = os.path.join(target_dir, t, "stdout.txt")

		try:
			with open(result_path, "rb") as result_file:
				result = pickle.load(result_file)
				if result.get("data_error", "") == "Insufficient Read Counts":
					insufficient_data_jobs.append(t)
					continue
				if result.get("data_error", "") == "Insufficient Markers":
					insufficient_snps_jobs.append(t)
					continue
		except (EOFError, IOError):
			failed_jobs.append(t)
			continue
		
		summary["names"].append(t)
		if "full" in model_flavors:
			summary["causal_sets_full"].append(result["causal_set_full"])
			summary["ppas_full"].append(result["ppas_full"])
			set_size = np.count_nonzero(result["causal_set_full"])
			set_prop = set_size / np.shape(result["causal_set_full"])[0]
			summary["set_sizes_full"].append(set_size)
			summary["set_props_full"].append(set_prop)
			for k in summary["thresholds_full"].keys():
				if set_size <= k:
					summary["thresholds_full"][k] += 1
			size_probs = np.resize(result["size_probs_full"], 6)
			summary["size_probs_full"] += size_probs
		if "indep" in model_flavors:
			summary["causal_sets_indep"].append(result["causal_set_indep"])
			summary["ppas_indep"].append(result["ppas_indep"])
			set_size = np.count_nonzero(result["causal_set_indep"])
			set_prop = set_size / np.shape(result["causal_set_indep"])[0]
			summary["set_sizes_indep"].append(set_size)
			summary["set_props_indep"].append(set_prop)
			for k in summary["thresholds_indep"].keys():
				if set_size <= k:
					summary["thresholds_indep"][k] += 1
			size_probs = np.resize(result["size_probs_indep"], 6)
			summary["size_probs_indep"] += size_probs
		if "eqtl" in model_flavors:
			summary["causal_sets_eqtl"].append(result["causal_set_eqtl"])
			summary["ppas_eqtl"].append(result["ppas_eqtl"])
			set_size = np.count_nonzero(result["causal_set_eqtl"])
			set_prop = set_size / np.shape(result["causal_set_eqtl"])[0]
			summary["set_sizes_eqtl"].append(set_size)
			summary["set_props_eqtl"].append(set_prop)
			for k in summary["thresholds_eqtl"].keys():
				if set_size <= k:
					summary["thresholds_eqtl"][k] += 1
			size_probs = np.resize(result["size_probs_eqtl"], 6)
			summary["size_probs_eqtl"] += size_probs
		if "ase" in model_flavors:
			summary["causal_sets_ase"].append(result["causal_set_ase"])
			summary["ppas_ase"].append(result["ppas_ase"])
			set_size = np.count_nonzero(result["causal_set_ase"])
			set_prop = set_size / np.shape(result["causal_set_ase"])[0]
			summary["set_sizes_ase"].append(set_size)
			summary["set_props_ase"].append(set_prop)
			for k in summary["thresholds_ase"].keys():
				if set_size <= k:
					summary["thresholds_ase"][k] += 1
			size_probs = np.resize(result["size_probs_ase"], 6)
			summary["size_probs_ase"] += size_probs
		if "acav" in model_flavors:
			summary["causal_sets_caviar_ase"].append(result["causal_set_caviar_ase"])
			summary["ppas_caviar_ase"].append(result["ppas_caviar_ase"])
			set_size = np.count_nonzero(result["causal_set_caviar_ase"])
			set_prop = set_size / np.shape(result["causal_set_caviar_ase"])[0]
			summary["set_sizes_caviar_ase"].append(set_size)
			summary["set_props_caviar_ase"].append(set_prop)
			for k in summary["thresholds_caviar_ase"].keys():
				if set_size <= k:
					summary["thresholds_caviar_ase"][k] += 1

		successes += 1

	with open(os.path.join(out_dir, "failed_jobs.txt"), "w") as fail_out:
		fail_out.write("\n".join(failed_jobs))

	with open(os.path.join(out_dir, "insufficient_data_jobs.txt"), "w") as insufficient_data_out:
		insufficient_data_out.write("\n".join(insufficient_data_jobs))

	with open(os.path.join(out_dir, "insufficient_snps_jobs.txt"), "w") as insufficient_snps_out:
		insufficient_snps_out.write("\n".join(insufficient_snps_jobs))
	
	write_thresholds(summary, out_dir, successes, model_flavors)
	write_size_probs(summary, out_dir, successes, model_flavors)
	plot_dist(summary, out_dir, name, model_flavors, "size", False)
	plot_dist(summary, out_dir, name, model_flavors, "prop", False)
	plot_dist(summary, out_dir, name, model_flavors, "size", True)
	plot_dist(summary, out_dir, name, model_flavors, "prop", True)

	return summary

def interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name, recall_model_flavors=None):
	if model_flavors == "all":
		model_flavors = set(["full", "indep", "eqtl", "ase", "acav"])

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	series = {}
	if "full" in model_flavors:
		series["avg_sets_full"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_full"] = {}
		series["all_props_full"] = {}
		series["recall_full"] = {}
	if "indep" in model_flavors:
		series["avg_sets_indep"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_indep"] = {}
		series["all_props_indep"] = {}
		series["recall_indep"] = {}
	if "eqtl" in model_flavors:
		series["avg_sets_eqtl"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_eqtl"] = {}
		series["all_props_eqtl"] = {}
		series["recall_eqtl"] = {}
	if "ase" in model_flavors:
		series["avg_sets_ase"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_ase"] = {}
		series["all_props_ase"] = {}
		series["recall_ase"] = {}
	if "acav" in model_flavors:
		series["avg_sets_caviar_ase"] = {i: 0 for i in primary_var_vals}
		series["all_sizes_caviar_ase"] = {}
		series["all_props_caviar_ase"] = {}
		series["recall_caviar_ase"] = {}

	sig_snps = {}
	num_sigs = {}
	for ind, val in enumerate(summaries[0]["ppas_eqtl"]):
		gene_name = summaries[0]["names"][ind]
		sigs = set([sind for sind, sval in enumerate(val) if sval >= 0.1])
		sig_snps[gene_name] = sigs
		num_sigs[gene_name] = len(sigs)

	for ind, val in enumerate(summaries):
		var_val = primary_var_vals[ind]
		data_size = sum([1 for i in val["names"] if num_sigs.get(i, 0) > 0])
		if "full" in model_flavors:
			series["avg_sets_full"][var_val] = np.mean(val["set_sizes_full"])
			series["all_sizes_full"][var_val] = val["set_sizes_full"]
			series["all_props_full"][var_val] = val["set_props_full"]
			series["recall_full"][var_val] = {}
			for sind, sval in enumerate(val["ppas_full"]):
				gene_name = val["names"][sind]
				if gene_name not in sigs:
					continue
				sigs = sig_snps[gene_name]
				num = num_sigs[gene_name]
				if num > 0:
					loc_size = len(sval)
					ppa_idx_sorted = sorted(range(loc_size), key=lambda x:sval[x], reverse=True)
					for xind, xval in enumerate(ppa_idx_sorted):
						if xval in sigs:
							pos = xind / loc_size
							series["recall_full"][var_val].setdefault(pos, 0)
							series["recall_full"][var_val][pos] += 1. / (num * data_size)
						
		if "indep" in model_flavors:
			series["avg_sets_indep"][var_val] = np.mean(val["set_sizes_indep"])
			series["all_sizes_indep"][var_val] = val["set_sizes_indep"]
			series["all_props_indep"][var_val] = val["set_props_indep"]
			series["recall_indep"][var_val] = {}
			for sind, sval in enumerate(val["ppas_indep"]):
				gene_name = val["names"][sind]
				if gene_name not in sigs:
					continue
				sigs = sig_snps[gene_name]
				num = num_sigs[gene_name]
				if num > 0:
					loc_size = len(sval)
					ppa_idx_sorted = sorted(range(loc_size), key=lambda x:sval[x], reverse=True)
					for xind, xval in enumerate(ppa_idx_sorted):
						if xval in sigs:
							pos = xind / loc_size
							series["recall_indep"][var_val].setdefault(pos, 0)
							series["recall_indep"][var_val][pos] += 1. / (num * data_size)

		if "eqtl" in model_flavors:
			series["avg_sets_eqtl"][var_val] = np.mean(val["set_sizes_eqtl"])
			series["all_sizes_eqtl"][var_val] = val["set_sizes_eqtl"]
			series["all_props_eqtl"][var_val] = val["set_props_eqtl"]
			series["recall_eqtl"][var_val] = {}
			for sind, sval in enumerate(val["ppas_eqtl"]):
				gene_name = val["names"][sind]
				if gene_name not in sigs:
					continue
				sigs = sig_snps[gene_name]
				num = num_sigs[gene_name]
				if num > 0:
					loc_size = len(sval)
					ppa_idx_sorted = sorted(range(loc_size), key=lambda x:sval[x], reverse=True)
					for xind, xval in enumerate(ppa_idx_sorted):
						if xval in sigs:
							pos = xind / loc_size
							series["recall_eqtl"][var_val].setdefault(pos, 0)
							series["recall_eqtl"][var_val][pos] += 1. / (num * data_size)

		if "ase" in model_flavors:
			series["avg_sets_ase"][var_val] = np.mean(val["set_sizes_ase"])
			series["all_sizes_ase"][var_val] = val["set_sizes_ase"]
			series["all_props_ase"][var_val] = val["set_props_ase"]
			series["recall_ase"][var_val] = {}
			for sind, sval in enumerate(val["ppas_ase"]):
				gene_name = val["names"][sind]
				if gene_name not in sigs:
					continue
				sigs = sig_snps[gene_name]
				num = num_sigs[gene_name]
				if num > 0:
					loc_size = len(sval)
					ppa_idx_sorted = sorted(range(loc_size), key=lambda x:sval[x], reverse=True)
					for xind, xval in enumerate(ppa_idx_sorted):
						if xval in sigs:
							pos = xind / loc_size
							series["recall_ase"][var_val].setdefault(pos, 0)
							series["recall_ase"][var_val][pos] += 1. / (num * data_size)

		if "acav" in model_flavors:
			series["avg_sets_caviar_ase"][var_val] = np.mean(val["set_sizes_caviar_ase"])
			series["all_sizes_caviar_ase"][var_val] = val["set_sizes_caviar_ase"]
			series["all_props_caviar_ase"][var_val] = val["set_props_caviar_ase"]
			series["recall_caviar_ase"][var_val] = {}
			for sind, sval in enumerate(val["ppas_caviar_ase"]):
				gene_name = val["names"][sind]
				if gene_name not in sigs:
					continue
				sigs = sig_snps[gene_name]
				num = num_sigs[gene_name]
				if num > 0:
					loc_size = len(sval)
					ppa_idx_sorted = sorted(range(loc_size), key=lambda x:sval[x], reverse=True)
					for xind, xval in enumerate(ppa_idx_sorted):
						if xval in sigs:
							pos = xind / loc_size
							series["recall_caviar_ase"][var_val].setdefault(pos, 0)
							series["recall_caviar_ase"][var_val][pos] += 1. / (num * data_size)

	if recall_model_flavors is None:
		recall_model_flavors = model_flavors
	plot_recall(series, primary_var_vals, primary_var_name, out_dir, name, recall_model_flavors)
	plot_series(series, primary_var_vals, primary_var_name, out_dir, name, model_flavors, "size")
	plot_series(series, primary_var_vals, primary_var_name, out_dir, name, model_flavors, "prop")

if __name__ == '__main__':
	# # Kidney Cancer

	# # Normal
	# model_flavors = set(["indep", "eqtl", "ase", "acav"])

	# # Normal, all samples
	# target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_all"
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_all"
	# name = "Kidney RNA-Seq\nAll Normal Samples"

	# normal_all = interpret(target_dir, out_dir, name, model_flavors)

	# # Normal, 50 samples
	# target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_50"
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_50"
	# name = "Kidney RNA-Seq\n50 Normal Samples"

	# normal_50 = interpret(target_dir, out_dir, name, model_flavors)

	# # Normal, 10 samples
	# target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_10"
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_10"
	# name = "Kidney RNA-Seq\n10 Normal Samples"

	# normal_10 = interpret(target_dir, out_dir, name, model_flavors)

	# # Normal, across sample sizes
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_sample_sizes"
	# name = "Kidney RNA-Seq, Normal Samples"
	# model_flavors = set(["indep", "eqtl", "acav"])
	# recall_model_flavors = set(["eqtl", "acav"])
	# summaries = [normal_all, normal_50, normal_10]
	# primary_var_vals = [70, 50, 10]
	# primary_var_name = "Sample Size"

	# interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name, recall_model_flavors=recall_model_flavors)

	# Tumor
	model_flavors = set(["indep", "eqtl", "ase", "acav"])

	# Tumor, all samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_all"
	name = "Kidney RNA-Seq\nAll Tumor Samples"

	tumor_all = interpret(target_dir, out_dir, name, model_flavors)

	# Tumor, 200 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_200"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_200"
	name = "Kidney RNA-Seq\n200 Tumor Samples"

	tumor_200 = interpret(target_dir, out_dir, name, model_flavors)

	# Tumor, 100 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_100"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_100"
	name = "Kidney RNA-Seq\n100 Tumor Samples"

	tumor_100 = interpret(target_dir, out_dir, name, model_flavors)

	# Tumor, 50 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_50"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_50"
	name = "Kidney RNA-Seq\n50 Tumor Samples"

	tumor_50 = interpret(target_dir, out_dir, name, model_flavors)

	# Tumor, 10 samples
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_10"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_10"
	name = "Kidney RNA-Seq\n10 Tumor Samples"

	tumor_10 = interpret(target_dir, out_dir, name, model_flavors)

	# Tumor, across sample sizes
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_sample_sizes"
	name = "Kidney RNA-Seq, Tumor Samples"
	model_flavors = set(["indep", "eqtl", "acav"])
	recall_model_flavors = set(["indep", "acav"])
	summaries = [tumor_all, tumor_200, tumor_100, tumor_50, tumor_10]
	primary_var_vals = [524, 200, 100, 50, 10]
	primary_var_name = "Sample Size"

	interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name, recall_model_flavors=recall_model_flavors)

	# # Tumor, low heritability, all samples
	# target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all_low_herit"
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_all_low_herit"
	# name = "Kidney RNA-Seq\nAll Tumor Samples"

	# tumor_low_herit = interpret(target_dir, out_dir, name, model_flavors)

	# #Prostate Cancer
	
	# # Normal
	# model_flavors = set(["indep", "eqtl", "ase", "acav"])

	# # Normal, all samples
	# target_dir = "/bcb/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_all"
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_normal_all"
	# name = "Prostate ChIP-Seq\nAll Normal Samples"

	# normal_all = interpret(target_dir, out_dir, name, model_flavors)

	# # Normal, 10 samples
	# target_dir = "/bcb/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_10"
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_normal_10"
	# name = "Prostate ChIP-Seq\n10 Normal Samples"

	# normal_10 = interpret(target_dir, out_dir, name, model_flavors)

	# # Normal, across sample sizes
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_normal_sample_sizes"
	# name = "Prostate ChIP-Seq, Normal Samples"
	# model_flavors = set(["indep", "eqtl", "acav"])
	# summaries = [normal_all, normal_10]
	# primary_var_vals = [24, 10]
	# primary_var_name = "Sample Size"

	# interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name)

	# # Tumor
	# model_flavors = set(["indep", "eqtl", "ase", "acav"])

	# # Tumor, all samples
	# target_dir = "/bcb/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_all"
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_tumor_all"
	# name = "Prostate ChIP-Seq\nAll Tumor Samples"

	# tumor_all = interpret(target_dir, out_dir, name, model_flavors)

	# # Tumor, 10 samples
	# target_dir = "/bcb/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_10"
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_tumor_10"
	# name = "Prostate ChIP-Seq\n10 Tumor Samples"

	# tumor_10 = interpret(target_dir, out_dir, name, model_flavors)

	# # Tumor, across sample sizes
	# out_dir = "/bcb/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_tumor_sample_sizes"
	# name = "Prostate ChIP-Seq, Tumor Samples"
	# model_flavors = set(["indep", "eqtl", "acav"])
	# summaries = [tumor_all, tumor_10]
	# primary_var_vals = [24, 10]
	# primary_var_name = "Sample Size"

	# interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name)

