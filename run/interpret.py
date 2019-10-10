import numpy as np
import os
import time
import sys
import traceback
import pickle
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

pal = sns.color_palette()
COLORMAP = {
	"full": pal[1],
	"indep": pal[3],
	"ase": pal[2],
	"acav": pal[5],
	"eqtl": pal[0],
	"cav": pal[7],
	"rasq": pal[6],
	"fmb": pal[8],
}
NAMEMAP = {
	"full": "PLASMA-J",
	"indep": "PLASMA-JI",
	"ase": "PLASMA-AS",
	"acav": "AS-Meta",
	"eqtl": "QTL-Only",
	"cav": "CAVIAR",
	"rasq": "RASQUAL+",
	"fmb": "FINEMAP",
}

def get_targets(list_path):
	with open(list_path, "rb") as list_file:
		targets = pickle.load(list_file)
	return targets

def write_thresholds(summary, out_dir, total_jobs, model_flavors):
	header = "\t".join(["Model"] + [str(i) for i in summary["thresholds"]]) + "\n"	
	thresholds_list = [header]
	for f in model_flavors:
		data = [NAMEMAP[f]] + [summary["thresholds_{0}".format(f)][i] for i in summary["thresholds"]]
		line = "\t".join([str(i) for i in data]) + "\n"
		thresholds_list.append(line)

	out_path = os.path.join(out_dir, "causal_set_thresholds.txt")
	with open(out_path, "w") as out_file:
		out_file.writelines(thresholds_list)

def write_size_probs(summary, out_dir, total_jobs, model_flavors):
	size_probs_list = []
	for f in model_flavors:
		data = "\t".join([NAMEMAP[f]] + [str(i) for i in summary["size_probs_{0}".format(f)] / total_jobs]) + "\n"
		size_probs_list.append(data)

	out_path = os.path.join(out_dir, "causal_set_size_probabilities.txt")
	with open(out_path, "w") as out_file:
		out_file.writelines(size_probs_list)

def write_sumstats(summary, out_dir, total_jobs, model_flavors):
	header = "\t".join(["Model", "Mean", "Variance", "25 Perc", "Median", "75 Perc"]) + "\n"
	sumstats_list = [header]
	for f in model_flavors:
		sizes = summary["set_sizes_{0}".format(f)]
		line = "\t".join([
			NAMEMAP[f], 
			str(np.nanmean(sizes)), 
			str(np.nanvar(sizes)), 
			str(np.nanpercentile(sizes, 25)),
			str(np.nanmedian(sizes)),
			str(np.nanpercentile(sizes, 75))
		]) + "\n"
		sumstats_list.append(line)

	out_path = os.path.join(out_dir, "causal_set_stats.txt")

	with open(out_path, "w") as out_file:
		out_file.writelines(sumstats_list)

def plot_violin(result, out_dir, name, model_flavors, metric):
	if metric == "size":
		kwd = "set_sizes"
		kwd_label = "Credible Set Size"
	elif metric == "prop":
		kwd = "set_props"
		kwd_label = "Credible Set Size (Proportion)"

	sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(4,2)})

	plt_data = []
	for f in model_flavors:
		set_sizes = result["{0}_{1}".format(kwd, f)]
		plt_data.extend([[i, NAMEMAP[f]] for i in set_sizes])

	labels = [kwd_label, "Model"]
	df = pd.DataFrame.from_records(plt_data, columns=labels)

	names = [NAMEMAP[m] for m in model_flavors]
	palette = [COLORMAP[m] for m in model_flavors]
	chart = sns.violinplot(
		x=kwd_label, 
		y="Model", 
		data=df, 
		order=names, 
		palette=palette,
		cut=0,
		scale="width",
	)
	plt.ylabel("")
	plt.title(name)
	if metric == "prop":
		plt.xlim(0, 1)
	elif metric == "size":
		plt.xlim(0, 500)
	plt.savefig(os.path.join(out_dir, "set_{0}_distribution.svg".format(metric)), bbox_inches='tight')
	plt.clf()

def plot_thresh(result, out_dir, name, model_flavors, total_jobs):
	sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(4,2)})

	threshs = result["thresholds"]
	plt_data = []
	for f in model_flavors:
		plt_data.append([1., np.inf, NAMEMAP[f]])
		thresh_data = result["thresholds_{0}".format(f)]
		for k, v in thresh_data.items():
			plt_data.append([v / total_jobs, k, NAMEMAP[f]])

	labels = ["Proportion of Loci", "Threshold", "Model"]
	df = pd.DataFrame.from_records(plt_data, columns=labels)

	names = [NAMEMAP[m] for m in model_flavors]
	palette = sns.cubehelix_palette(len(threshs) + 1, rot=-.25, light=.7)
	df_thresh = df.loc[df["Threshold"] == np.inf]
	chart = sns.barplot(
		x="Proportion of Loci", 
		y="Model", 
		data=df_thresh, 
		order=names, 
		color=palette[0], 
		ci=None
	)
	for i, t in enumerate(reversed(threshs)):
		df_thresh = df.loc[df["Threshold"] == t]
		chart = sns.barplot(
			x="Proportion of Loci", 
			y="Model", 
			data=df_thresh, 
			order=names, 
			color=palette[i+1], 
			ci=None
		)

	last_marker = [None for _ in range(len(model_flavors))]
	for i, f in enumerate(model_flavors):
		thresh_data = result["thresholds_{0}".format(f)]
		for k, v in sorted(thresh_data.items()):
			xval = v / total_jobs
			if (last_marker[i] is None and xval >= 0.04) or (last_marker[i] and (xval - last_marker[i]) >= 0.08):
				plt.text(
					xval,
					i,
					k,
					size="xx-small",
					weight="medium",
					ha="center",
					va="center",
					bbox={"boxstyle":"round", "pad":.25, "fc":"white", "ec":"white"}
				)
				last_marker[i] = xval

	plt.ylabel("")
	plt.title(name)
	plt.savefig(os.path.join(out_dir, "set_size_thresh.svg"), bbox_inches='tight')
	plt.clf()

def plot_dist(result, out_dir, name, model_flavors, metric, cumu):
	if metric == "size":
		kwd = "set_sizes"
	elif metric == "prop":
		kwd = "set_props"

	sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(4,2)})

	for f in model_flavors:
		set_sizes = result["{0}_{1}".format(kwd, f)]
		try:
			sns.distplot(
				set_sizes,
				hist=False,
				kde=True,
				kde_kws={"linewidth": 2, "shade":False, "cumulative":cumu},
				label=NAMEMAP[f],
				color=COLORMAP[f]
			)
		except Exception as e:
			trace = traceback.format_exc()
			print(trace, file=sys.stderr)

	if metric == "prop":
		plt.xlim(0, 1)
	elif metric == "size":
		plt.xlim(0, 600)
	plt.legend(title="Model")
	if cumu:
		cumu_kwd = "Cumulative "
		cumu_fname = "_cumu"
		yax = "Proportion of Markers"
	else:
		cumu_kwd = ""
		cumu_fname = ""
		yax = "Density"
	if metric == "size":
		plt.xlabel("Credible Set Size")
		plt.ylabel(yax)
		plt.title(name)
		plt.savefig(os.path.join(out_dir, "set_size_distribution{0}.svg".format(cumu_fname)))
	elif metric == "prop":
		plt.xlabel("Credible Set Size (Proportion)")
		plt.ylabel(yax)
		plt.title(name)
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
	for key, val in series.items():
		for f in model_flavors:
			for skey, sval in series["{0}_{1}".format(kwd, f)].items():
				for i in sval:
					dflst.append([i, skey, NAMEMAP[f]])
	res_df = pd.DataFrame(dflst, columns=[label, primary_var_name, "Model"])

	names = [NAMEMAP[m] for m in model_flavors]
	palette = [COLORMAP[m] for m in model_flavors]
	
	sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(9,3)})
	# plt.rcParams["figure.figsize"] = (9,3) ####

	if metric == "prop":
		plt.ylim(0, 1)
	elif metric == "size":
		plt.ylim(0, 500)
	g = sns.violinplot(
		x=primary_var_name,
		y=label,
		hue="Model",
		data=res_df,
		order=primary_var_vals,
		hue_order=names,
		palette=palette,
		scale="width",
		cut=0
	)
	g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(9, 3) ####
	plt.title(name)
	plt.savefig(os.path.join(out_dir, "{0}_violin.svg".format(filename)))
	plt.clf()

	g = sns.barplot(
		x=primary_var_name, 
		y=label,
		hue="Model",
		data=res_df,
		order=primary_var_vals,
		hue_order=names,
		palette=palette
	)
	g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
	plt.title(name)
	plt.savefig(os.path.join(out_dir, "{0}_bar.svg".format(filename)))
	plt.clf()

def plot_recall(series, primary_var_vals, primary_var_name, out_dir, name, model_flavors):
	# print(model_flavors) ####
	# print(series.keys()) ####
	dflst = []
	for key, val in series.items():
		for f in model_flavors:
			for skey, sval in series["recall_{0}".format(f)].items():
				data = sorted(list(sval.items()), key=lambda x: x[0])
				cumu_recall = 0.
				for x, val in data:
					cumu_recall += val
					dflst.append([x, cumu_recall, skey, NAMEMAP[f]])
		
	labels = ["Proportion of Selected Markers", "Inclusion Rate", primary_var_name, "Model"]
	res_df = pd.DataFrame(dflst, columns=labels)
	# print(res_df) ####
		
	sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(6,5)})
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
	plt.title(name)
	plt.savefig(os.path.join(out_dir, "inclusion.svg"))
	plt.clf()

	for f in model_flavors:
		title = "{0}, {1} Model".format(name, NAMEMAP[f])
		sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(6,5)})
		palette = sns.cubehelix_palette(len(primary_var_vals))
		sns.lineplot(
			x="Proportion of Selected Markers",
			y="Inclusion Rate",
			hue=primary_var_name,
			data=res_df.query("Model == '{0}'".format(NAMEMAP[f])),
			sort=True,
			palette=palette
		)
		plt.title(title)
		plt.savefig(os.path.join(out_dir, "inclusion_{0}.svg".format(f)))
		plt.clf()


def interpret(targets, target_dir, out_dir, name, model_flavors, thresholds, fail_list_out=None, sig_filter=None):
	if model_flavors == "all":
		model_flavors = ["indep", "full", "ase", "acav", "eqtl", "fmb"]

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if targets == "all":
		targets = os.listdir(target_dir)

	summary = {"names":[], "thresholds": thresholds}
	for f in model_flavors:
		summary["causal_sets_{0}".format(f)] = []
		summary["ppas_{0}".format(f)] = []
		summary["set_sizes_{0}".format(f)] = []
		summary["set_props_{0}".format(f)] = []
		summary["thresholds_{0}".format(f)] = {i : 0 for i in thresholds}
		summary["size_probs_{0}".format(f)] = np.zeros(6)

	failed_jobs = []
	insufficient_data_jobs = []
	insufficient_snps_jobs = []
	non_sig_jobs = []
	successes = 0

	# causal_zscores_fmb = [] ####
	# small_loci = set([ ####
	# 	"ENSG00000134996.11",
	# 	"ENSG00000137364.4",
	# 	"ENSG00000160957.8",
	# 	"ENSG00000175387.11",
	# 	"ENSG00000134575.5",
	# 	"ENSG00000156968.8",
	# 	"ENSG00000157111.8",
	# 	"ENSG00000158258.11",
	# 	"ENSG00000157045.4",
	# 	"ENSG00000170222.11",
	# 	"ENSG00000161036.6",
	# 	"ENSG00000175198.10",
	# 	"ENSG00000140105.13",
	# 	"ENSG00000148339.8",
	# 	"ENSG00000160953.10",
	# 	"ENSG00000168589.10",
	# 	"ENSG00000173566.9",
	# 	"ENSG00000176842.10",
	# 	"ENSG00000167701.9",
	# 	"ENSG00000182095.10",
	# ])
	# print("") ####
	# small_sets = [] ####

	for t in targets:
		# print(t) ####
		result_path = os.path.join(target_dir, t, "output.pickle")
		# stdout_path = os.path.join(target_dir, t, "stdout.txt")

		try:
			with open(result_path, "rb") as result_file:
				result = pickle.load(result_file, encoding='latin1')
			if result.get("data_error", "") == "Insufficient Read Counts":
				insufficient_data_jobs.append(t)
				continue
			if result.get("data_error", "") == "Insufficient Markers":
				insufficient_snps_jobs.append(t)
				continue
			if "run_error" in result:
				failed_jobs.append(t)
				continue

			if sig_filter is not None:
				if np.abs(np.nanmax(result["z_beta"])) < sig_filter:
					non_sig_jobs.append(t)
					continue

			# if np.sum(result["causal_set_indep"]) <= 10: ####
			# 	small_sets.append([t, np.sum(result["causal_set_indep"].astype(int))]) ####
			# 	print("{0}\t{1}".format(t, np.sum(result["causal_set_indep"].astype(int)))) ####
			# zb = np.full(np.shape(result["causal_set_fmb"]), np.nan) ####
			# np.put(zb, result["informative_snps"], result["z_beta"]) ####
			# causal_zscores_fmb.append(zb[result["causal_set_fmb"].astype(bool)]) ####
			# causal_zscores_fmb.append([np.sum(result["causal_set_fmb"]), t]) ####
			# if t in small_loci:
			# 	res = [
			# 		t, 
			# 		np.sum(result["causal_set_fmb"]),
			# 		np.mean(result["causal_set_fmb"]), 
			# 		np.amax(np.abs(result["z_beta"])), 
			# 		np.sum(result["causal_set_ase"]), 
			# 		np.mean(result["causal_set_ase"]), 
			# 		np.amax(np.abs(result["z_phi"]))
			# 	]
				# causal_zscores_fmb.append(res) ####

		except (EOFError, IOError):
			failed_jobs.append(t)
			continue

		# try: ####
		# 	r = result["causal_set_fmb"]
		# except Exception: ####
		# 	print(t)
		# 	print(result.keys())
		# 	# raise
		# 	continue
		
		# print(result_path) ####
		# print(result.keys()) ####
		summary["names"].append(t)
		for f in model_flavors:
			summary["causal_sets_{0}".format(f)].append(result["causal_set_{0}".format(f)])
			summary["ppas_{0}".format(f)].append(result["ppas_{0}".format(f)])
			set_size = np.count_nonzero(result["causal_set_{0}".format(f)])
			set_prop = set_size / np.shape(result["causal_set_{0}".format(f)])[0]
			summary["set_sizes_{0}".format(f)].append(set_size)
			summary["set_props_{0}".format(f)].append(set_prop)
			for k in list(summary["thresholds_{0}".format(f)].keys()):
				if set_size <= k:
					summary["thresholds_{0}".format(f)][k] += 1
			size_probs = np.resize(result["size_probs_{0}".format(f)], 6)
			summary["size_probs_{0}".format(f)] += size_probs

		successes += 1

	# print(sorted(causal_zscores_fmb, key=np.size)[:20]) ####
	# lst = sorted(causal_zscores_fmb, key=lambda x:x[0])[:20] ####
	# for i in lst:
	# 	print("\t".join([str(j) for j in i])) ####
	# # print("")
	# small_sets.sort(key=lambda x:x[1]) ####
	# with open(os.path.join(out_dir, "small_sets.txt"), "w") as out_file: ####
	# 	for i in small_sets: ####
	# 		out_file.writelines("{0}\t{1}\n".format(*i)) ####

	if fail_list_out is not None:
		with open(fail_list_out, "wb") as fail_list_file:
			pickle.dump(failed_jobs, fail_list_file)

	with open(os.path.join(out_dir, "failed_jobs.txt"), "w") as fail_out:
		fail_out.write("\n".join(failed_jobs) + "\n")

	with open(os.path.join(out_dir, "insufficient_data_jobs.txt"), "w") as insufficient_data_out:
		insufficient_data_out.write("\n".join(insufficient_data_jobs) + "\n")

	with open(os.path.join(out_dir, "insufficient_snps_jobs.txt"), "w") as insufficient_snps_out:
		insufficient_snps_out.write("\n".join(insufficient_snps_jobs) + "\n")

	write_thresholds(summary, out_dir, successes, model_flavors)
	write_size_probs(summary, out_dir, successes, model_flavors)
	write_sumstats(summary, out_dir, successes, model_flavors)
	plot_thresh(summary, out_dir, name, model_flavors, successes)
	plot_violin(summary, out_dir, name, model_flavors, "size")
	plot_violin(summary, out_dir, name, model_flavors, "prop")
	plot_dist(summary, out_dir, name, model_flavors, "size", True)
	plot_dist(summary, out_dir, name, model_flavors, "prop", True)

	print(successes)

	return summary

def interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name, recall_model_flavors=None):
	if model_flavors == "all":
		model_flavors = ["indep", "full", "ase", "acav", "eqtl", "fmb"]

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	series = {}
	for f in model_flavors:
		series["avg_sets_{0}".format(f)] = {i: 0 for i in primary_var_vals}
		series["all_sizes_{0}".format(f)] = {}
		series["all_props_{0}".format(f)] = {}
		series["recall_{0}".format(f)] = {}

	sig_snps = {}
	num_sigs = {}
	for ind, val in enumerate(summaries[0]["ppas_fmb"]):
		gene_name = summaries[0]["names"][ind]
		sigs = set([sind for sind, sval in enumerate(val) if sval >= 0.1])
		sig_snps[gene_name] = sigs
		num_sigs[gene_name] = len(sigs)

	for ind, val in enumerate(summaries):
		var_val = primary_var_vals[ind]
		data_size = sum([1 for i in val["names"] if num_sigs.get(i, 0) > 0])
		for f in model_flavors:
			series["avg_sets_{0}".format(f)][var_val] = np.mean(val["set_sizes_{0}".format(f)])
			series["all_sizes_{0}".format(f)][var_val] = val["set_sizes_{0}".format(f)]
			series["all_props_{0}".format(f)][var_val] = val["set_props_{0}".format(f)]
			series["recall_{0}".format(f)][var_val] = {}
			for sind, sval in enumerate(val["ppas_{0}".format(f)]):
				gene_name = val["names"][sind]
				if gene_name not in sig_snps:
					continue
				sigs = sig_snps[gene_name]
				num = num_sigs[gene_name]
				if num > 0:
					loc_size = len(sval)
					ppa_idx_sorted = sorted(list(range(loc_size)), key=lambda x:sval[x], reverse=True)
					for xind, xval in enumerate(ppa_idx_sorted):
						if xval in sigs:
							pos = xind / loc_size
							series["recall_{0}".format(f)][var_val].setdefault(pos, 0)
							series["recall_{0}".format(f)][var_val][pos] += 1. / (num * data_size)

	# print(series.keys()) ####
	if recall_model_flavors is None:
		recall_model_flavors = model_flavors
	plot_series(series, primary_var_vals, primary_var_name, out_dir, name, model_flavors, "size")
	plot_series(series, primary_var_vals, primary_var_name, out_dir, name, model_flavors, "prop")
	plot_recall(series, primary_var_vals, primary_var_name, out_dir, name, recall_model_flavors)
	
if __name__ == '__main__':
	thresholds = [1, 5, 10, 20, 50, 100]
	
	# Kidney Cancer

	# # Normal
	# model_flavors = ["indep", "ase", "acav", "fmb",]
	# targets = get_targets("/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/normal_fdr05.pickle")

	# # Normal, all samples
	# target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_all"
	# out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_all"
	# name = "Kidney RNA-Seq, All Normal Samples"

	# normal_all = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# # Normal, 50 samples
	# target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_50"
	# out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_50"
	# name = "Kidney RNA-Seq, 50 Normal Samples"

	# normal_50 = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# # Normal, 10 samples
	# target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_normal_10"
	# out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_10"
	# name = "Kidney RNA-Seq, 10 Normal Samples"

	# normal_10 = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# # Normal, across sample sizes
	# out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_sample_sizes"
	# name = "Kidney RNA-Seq, Normal Samples"
	# model_flavors = ["indep", "ase", "acav", "fmb",]
	# recall_model_flavors = ["indep", "acav"]
	# summaries = [normal_all, normal_50, normal_10]
	# primary_var_vals = [70, 50, 10]
	# primary_var_name = "Sample Size"

	# interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name, recall_model_flavors=recall_model_flavors)

	# Tumor
	model_flavors = ["indep", "ase", "acav", "fmb",]
	targets = get_targets("/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/tumor_fdr05.pickle")

	# Tumor, all samples
	target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_all"
	name = "Kidney RNA-Seq, All Tumor Samples"

	tumor_all = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# Tumor, all samples, presentation
	target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all"
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_all_pres"
	name = ""

	tumor_all_pres = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# Tumor, 200 samples
	target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_200"
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_200"
	name = "Kidney RNA-Seq, 200 Tumor Samples"

	tumor_200 = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# Tumor, 100 samples
	target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_100"
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_100"
	name = "Kidney RNA-Seq, 100 Tumor Samples"

	tumor_100 = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# Tumor, 50 samples
	target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_50"
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_50"
	name = "Kidney RNA-Seq, 50 Tumor Samples"

	tumor_50 = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# Tumor, 10 samples
	target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_10"
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_10"
	name = "Kidney RNA-Seq, 10 Tumor Samples"

	tumor_10 = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# Tumor, across sample sizes
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_sample_sizes"
	name = "Kidney RNA-Seq, Tumor Samples"
	model_flavors = ["indep", "ase", "acav", "fmb",]
	recall_model_flavors = ["indep", "acav"]
	summaries = [tumor_all, tumor_200, tumor_100, tumor_50, tumor_10]
	primary_var_vals = [524, 200, 100, 50, 10]
	primary_var_name = "Sample Size"

	interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name, recall_model_flavors=recall_model_flavors)

	# # Tumor, low heritability, all samples
	# model_flavors = ["indep", "ase", "acav", "fmb",]
	# targets = get_targets("/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/tumor_fdr05.pickle")

	# target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_all_low_herit"
	# out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_all_low_herit"
	# name = "Kidney RNA-Seq, All Tumor Samples"

	# tumor_low_herit = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# Kidney Cancer, Multi CV

	# # Normal, all samples
	# model_flavors = ["indep", "ase", "fmb"]
	# targets = get_targets("/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/normal_fdr05.pickle")

	# target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/shotgun_normal_all"
	# out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/shotgun_normal_all"
	# fail_list_out = "/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/shotgun_normal_fail.pickle"
	# name = "Kidney RNA-Seq, All Normal Samples"

	# normal_multi_cv = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds, fail_list_out=fail_list_out)

	# # Tumor, all samples
	# model_flavors = ["indep", "ase", "fmb"]
	# targets = get_targets("/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/tumor_fdr05.pickle")

	# target_dir = "/agusevlab/awang/job_data/KIRC_RNASEQ/outs/shotgun_tumor_all"
	# out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/shotgun_tumor_all"
	# fail_list_out = "/agusevlab/awang/job_data/KIRC_RNASEQ/gene_lists/shotgun_tumor_fail.pickle"
	# name = "Kidney RNA-Seq, All Tumor Samples"

	# tumor_multi_cv = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds, fail_list_out=fail_list_out)

	# # Prostate Cancer

	thresholds = [20, 50, 100, 200, 300, 400]
	
	# # Normal
	# model_flavors = ["indep", "ase", "acav", "fmb",]
	# targets = "all"

	# # Normal, all samples
	# target_dir = "/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_all"
	# out_dir = "/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_normal_all"
	# name = "Prostate ChIP-Seq, All Normal Samples"

	# normal_all = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# # Normal, 10 samples
	# target_dir = "/agusevlab/awang/job_data/prostate_chipseq_normal/outs/1cv_normal_10"
	# out_dir = "/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_normal_10"
	# name = "Prostate ChIP-Seq, 10 Normal Samples"

	# normal_10 = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# # Normal, across sample sizes
	# out_dir = "/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_normal_sample_sizes"
	# name = "Prostate ChIP-Seq, Normal Samples"
	# model_flavors = ["indep", "ase", "acav", "fmb",]
	# summaries = [normal_all, normal_10]
	# primary_var_vals = [24, 10]
	# primary_var_name = "Sample Size"

	# interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name)

	# Tumor
	model_flavors = ["indep", "ase", "acav", "fmb",]
	targets = "all"

	# Tumor, all samples
	target_dir = "/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_all"
	out_dir = "/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_tumor_all"
	name = "Prostate ChIP-Seq, All Tumor Samples"

	tumor_all = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# Tumor, all samples, presentation
	target_dir = "/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_all"
	out_dir = "/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_tumor_all_pres"
	name = ""

	tumor_all = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# # Tumor, 10 samples
	# target_dir = "/agusevlab/awang/job_data/prostate_chipseq_tumor/outs/1cv_tumor_10"
	# out_dir = "/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_tumor_10"
	# name = "Prostate ChIP-Seq, 10 Tumor Samples"

	# tumor_10 = interpret(targets, target_dir, out_dir, name, model_flavors, thresholds)

	# # Tumor, across sample sizes
	# out_dir = "/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_tumor_sample_sizes"
	# name = "Prostate ChIP-Seq, Tumor Samples"
	# model_flavors = ["indep", "ase", "acav", "fmb",]
	# summaries = [tumor_all, tumor_10]
	# primary_var_vals = [24, 10]
	# primary_var_name = "Sample Size"

	# interpret_series(out_dir, name, model_flavors, summaries, primary_var_vals, primary_var_name)

