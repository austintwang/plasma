from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import os
import subprocess
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

def parse_output(s_out, lst_out, model_name):
	lines = s_out.strip().split("\n")
	for l in lines:
		cols = l.split("\t")
		entry = [model_name, float(cols[1]), float(cols[2]), float(cols[3]), float(cols[4]), -np.log10(float(cols[5]))]
		lst_out.append(entry)

def run_enrichment(bed_path_base, annot_path, script_path, model_flavors):
	if model_flavors == "all":
		model_flavors = set(["full", "indep", "eqtl", "ase", "acav"])

	lst_out = []
	if "full" in model_flavors:
		bed_path = bed_path_base.format("full")
		s_args = [script_path, bed_path, annot_path]
		s_out = subprocess.check_output(s_args)
		parse_output(s_out, lst_out, "Joint-Correlated")

	if "indep" in model_flavors:
		bed_path = bed_path_base.format("indep")
		s_args = [script_path, bed_path, annot_path]
		s_out = subprocess.check_output(s_args)
		parse_output(s_out, lst_out, "Joint-Independent")

	if "eqtl" in model_flavors:
		bed_path = bed_path_base.format("eqtl")
		s_args = [script_path, bed_path, annot_path]
		s_out = subprocess.check_output(s_args)
		parse_output(s_out, lst_out, "eQTL-Only")

	if "ase" in model_flavors:
		bed_path = bed_path_base.format("ase")
		s_args = [script_path, bed_path, annot_path]
		s_out = subprocess.check_output(s_args)
		parse_output(s_out, lst_out, "ASE-Only")

	if "acav" in model_flavors:
		bed_path = bed_path_base.format("acav")
		s_args = [script_path, bed_path, annot_path]
		s_out = subprocess.check_output(s_args)
		parse_output(s_out, lst_out, "CAVIAR-ASE")

	cols_out = [
		"Model", 
		"Minimum Posterior Probability", 
		"Odds Ratio", 
		"95% Confidence Interval Lower Bound", 
		"95% Confidence Interval Upper Bound", 
		"$-log_{10}$ p-Value"
	]

	df_out = pd.DataFrame(lst_out, columns=cols_out)

	return df_out

def plot_enrichment(out_dir, df_out, title):
	sns.set(style="whitegrid", font="Roboto")

	sns.barplot(
		x="Minimum Posterior Probability", 
		y="Odds Ratio",
		hue="Model",
		data=df_out
	)
	plt.title(title + "\nOdds Ratios")
	plt.savefig(os.path.join(out_dir, "enrichment_odds.svg"))
	plt.clf()

	sns.barplot(
		x="Minimum Posterior Probability", 
		y="$-log_{10}$ p-Value",
		hue="Model",
		data=df_out
	)
	plt.title(title + "\n$-log_{10}$ p-Values")
	plt.savefig(os.path.join(out_dir, "enrichment_pvals.svg"))
	plt.clf()

def enrichment(bed_path_base, annot_path, script_path, out_dir, title, model_flavors):
	df_out = run_enrichment(bed_path_base, annot_path, script_path, model_flavors)
	plot_enrichment(out_dir, df_out, title)
	data_path = os.path.join(out_dir, "enrichment_data.txt")
	df_out.to_csv(data_path, sep=str("\t"))


if __name__ == '__main__':
	enrichment_path = "/bcb/agusevlab/awang/job_data/enrichment"
	script_path = os.path.join(enrichment_path, "pct.sh")

	# Kidney Data
	annot_path = os.path.join(enrichment_path, "KIDNEY_DNASE.E086-DNase.imputed.narrowPeak.bed")
	model_flavors = set(["indep", "eqtl", "ase", "acav"])

	# Normal
	bed_path_base = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_normal_all/ldsr_{0}.bed"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_enrichment"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	title = "Kidney RNA-Seq, Open Chromatin Enrichment, Normal Samples"

	enrichment(bed_path_base, annot_path, script_path, out_dir, title, model_flavors)

	# Tumor
	bed_path_base = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_tumor_all/ldsr_{0}.bed"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_enrichment"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	title = "Kidney RNA-Seq, Open Chromatin Enrichment, Normal Samples"

	enrichment(bed_path_base, annot_path, script_path, out_dir, title, model_flavors)

	# Prostate Data
	annot_path = os.path.join(enrichment_path, "PRCA_HICHIP.MERGED_Annotated_FDR0.01_ncounts10.E.bed")
	model_flavors = set(["indep", "eqtl", "ase", "acav"])

	# Normal
	bed_path_base = "/bcb/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_normal_all/ldsr_{0}.bed"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_normal_enrichment"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	title = "Prostate ChIP-Seq, Chromatin Looping Enrichment, Normal Samples"

	enrichment(bed_path_base, annot_path, script_path, out_dir, title, model_flavors)

	# Tumor
	bed_path_base = "/bcb/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_tumor_all/ldsr_{0}.bed"
	out_dir = "/bcb/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_tumor_enrichment"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	title = "Prostate ChIP-Seq, Chromatin Looping Enrichment, Tumor Samples"

	enrichment(bed_path_base, annot_path, script_path, out_dir, title, model_flavors)
