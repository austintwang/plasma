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
import scipy.stats

try:
	import cPickle as pickle
except ImportError:
	import pickle


#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import sys
import traceback

try:
	import cPickle as pickle
except ImportError:
	import pickle


def region_plotter():

def plot_manhattan(pp_df, gene_name, out_dir):
	sns.set(style="whitegrid", font="Roboto")

	g = sns.FacetGrid(
		pp_df, 
		row="Sample Size", 
		col="Statistic", 
		hue="Causal",
		margin_titles=True, 
		height=1.7, 
		aspect=3
	)
	g.set(xticklabels=[])
	g.map(
		sns.scatterplot, 
		"Position", 
		"-log_10 p-Value", 
		color=".3", 
		linewidth=0,
	)

	axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

	# for ax in g.axes.flat:
	# 	labels = ["" for i in ax.get_xticklabels()] 
	# 	ax.set_xticklabels(labels) 

	plt.subplots_adjust(top=0.9)
	g.fig.suptitle("Association Statistics for {0}".format(gene_name))
	plt.savefig(os.path.join(out_dir, "manhattan_{0}.svg".format(gene_name)))
	plt.clf()

def manhattan(res_paths, sample_sizes, gene_name, causal_snps, out_dir):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	pp_lst = []
	for ind, val in enumerate(res_paths):
		with open(os.path.join(val, "output.pickle")) as res_file:
			result = pickle.load(res_file)
		with open(os.path.join(val, "input.pickle")) as inp_file:
			inputs = pickle.load(inp_file)

		snp_ids = inputs["snp_ids"]
		snp_pos = inputs["snp_pos"]

		causal_inds = set([ind for ind, val in enumerate(inputs["snp_ids"]) if val in causal_snps])

		for i, z in enumerate(result["z_phi"]):
			l = -np.log10(scipy.stats.norm.sf(abs(z))*2)
			causal = (i in causal_inds)
			info = [snp_pos[i], l, "AS", sample_sizes[ind], causal]
			pp_lst.append(info)

		for i, z in enumerate(result["z_beta"]):
			l = -np.log10(scipy.stats.norm.sf(abs(z))*2)
			causal = (i in causal_inds)
			info = [snp_pos[i], l, "QTL", sample_sizes[ind], causal]
			pp_lst.append(info)


	pp_cols = [
		"Position", 
		"-log_10 p-Value", 
		"Statistic", 
		"Sample Size",
		"Causal"
	]

	pp_df = pd.DataFrame(pp_lst, columns=pp_cols)

	plot_manhattan(pp_df, gene_name, out_dir)

if __name__ == '__main__':
	path_base = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/outs/1cv_tumor_{0}/{1}"

	# SCARB1
	res_paths = [path_base.format(i, "ENSG00000073060.11") for i in ["all", "200", "100", "50"]]
	sample_sizes = [524, 200, 100, 50]
	gene_name = "SCARB1"
	causal_snps = set(["rs4765621", "rs4765623"])
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/manhattan"

	manhattan(res_paths, sample_sizes, gene_name, causal_snps, out_dir)

	# DPF3
	res_paths = [path_base.format(i, "ENSG00000205683.7") for i in ["all", "200", "100", "50"]]
	sample_sizes = [524, 200, 100, 50]
	gene_name = "DPF3"
	causal_snps = set(["rs4903064"])
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/manhattan"
	manhattan(res_paths, sample_sizes, gene_name, causal_snps, out_dir)

	# GRAMD4
	res_paths = [path_base.format(i, "ENSG00000075240.12") for i in ["all", "200", "100", "50"]]
	sample_sizes = [524, 200, 100, 50]
	gene_name = "GRAMD4"
	causal_snps = set()
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/manhattan"
	manhattan(res_paths, sample_sizes, gene_name, causal_snps, out_dir)

