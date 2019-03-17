from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 

try:
	import cPickle as pickle
except ImportError:
	import pickle

def plot_manhattan(pp_df, gene_name, out_dir):
	sns.set(style="whitegrid", font="Roboto")

	g = sns.FacetGrid(pp_df, row="Sample Size", col="Model", margin_titles=True, height=1.7, aspect=3)
	g.set(xticklabels=[])
	g.map(
		sns.scatterplot, 
		"Marker", 
		"Posterior Probability", 
		color=".3", 
	)

	for ax in g.axes.flat:
		labels = ["" for i in ax.get_xticklabels()] 
		ax.set_xticklabels(labels) 

	plt.title("SNP Posterior Probabilites for {0}".format(gene_name))
	plt.savefig(os.path.join(out_dir, "manhattan_{0}.svg".format(gene_name)))
	plt.clf()

def manhattan(res_paths, sample_sizes, gene_name, out_dir, model_flavors):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	if model_flavors == "all":
		model_flavors = set(["full", "indep", "eqtl", "ase", "acav"])

	pp_lst = []
	for ind, val in enumerate(res_paths):
		with open(val) as res_file:
			result = pickle.load(res_file)

		if "full" in model_flavors:
			for i, p in enumerate(result["ppas_full"]):
				info = [i, p, "Joint-Correlated", sample_sizes[ind]]
				pp_lst.append(info)

		if "indep" in model_flavors:
			for i, p in enumerate(result["ppas_indep"]):
				info = [i, p, "Joint-Independent", sample_sizes[ind]]
				pp_lst.append(info)

		if "eqtl" in model_flavors:
			for i, p in enumerate(result["ppas_eqtl"]):
				info = [i, p, "eQTL-Only", sample_sizes[ind]]
				pp_lst.append(info)

		if "ase" in model_flavors:
			for i, p in enumerate(result["ppas_ase"]):
				info = [i, p, "ASE-Only", sample_sizes[ind]]
				pp_lst.append(info)

		if "acav" in model_flavors:
			for i, p in enumerate(result["ppas_acav"]):
				info = [i, p, "CAVIAR-ASE", sample_sizes[ind]]
				pp_lst.append(info)

	cols_out = [
		"Marker", 
		"Posterior Probability", 
		"Model", 
		"Sample Size",
	]

	pp_df = pd.DataFrame(pp_lst, columns=pp_cols)

	plot_manhattan(pp_df, gene_name, out_dir)

if __name__ == '__main__':
	res_path_base = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_{0}/{1}"

	# SCARB1
	res_paths = [res_path_base.format(i, "ENSG00000073060.11") for i in ["all", "200", "100", "50", "10"]]
	sample_sizes = [524, 200, 100, 50, 10]
	gene_name = "SCARB1"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/manhattan"
	model_flavors = set(["indep", "eqtl"])
	manhattan(res_paths, sample_sizes, gene_name, out_dir, model_flavors)

	# DPF3
	res_paths = [res_path_base.format(i, "ENSG00000205683.7") for i in ["all", "200", "100", "50", "10"]]
	sample_sizes = [524, 200, 100, 50, 10]
	gene_name = "DPF3"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/manhattan"
	model_flavors = set(["indep", "eqtl"])
	manhattan(res_paths, sample_sizes, gene_name, out_dir, model_flavors)

	# GRAMD4
	res_paths = [res_path_base.format(i, "ENSG00000075240.12") for i in ["all", "200", "100", "50", "10"]]
	sample_sizes = [524, 200, 100, 50, 10]
	gene_name = "GRAMD4"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/manhattan"
	model_flavors = set(["indep", "eqtl"])
	manhattan(res_paths, sample_sizes, gene_name, out_dir, model_flavors)

