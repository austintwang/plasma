import os
import subprocess
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import pickle

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
	"full": "PLASMA-JC",
	"indep": "PLASMA-J",
	"ase": "PLASMA-AS",
	"acav": "AS-Meta",
	"eqtl": "QTL-Only",
	"cav": "CAVIAR",
	"rasq": "RASQUAL+",
	"fmb": "FINEMAP",
}

COLORMAP_PRES = {
	"indep": pal[0],
	"acav": pal[2],
	"rasq": pal[1],
	"fmb": pal[3],
}
NAMEMAP_PRES = {
	"indep": "PLASMA",
	"acav": "AS-Meta",
	"rasq": "RASQUAL+",
	"fmb": "FINEMAP",
}

def fisher_enr(arg1, arg2, arg3, arg4):
	table = np.array([[arg1, arg2-arg1],[arg3-arg1, arg4-arg2-arg3+arg1]])
	return scipy.stats.fisher_exact(table)

def parse_output(s_out, lst_out, model_name):
	lines = s_out.decode("utf-8").strip().split("\n")
	for l in lines:
		cols = l.split("\t")
		odds, pval = fisher_enr(float(cols[2]), float(cols[3]), float(cols[4]), float(cols[5]))
		entry = [model_name, float(cols[1]), odds, -np.log10(pval)]
		lst_out.append(entry)

def run_enrichment(bed_path_base, annot_path, script_path, ctrl_path, model_flavors, presentation):
	if model_flavors == "all":
		model_flavors = ["full", "indep", "eqtl", "ase", "acav"]

	if presentation:
		namemap = NAMEMAP_PRES
	else:
		namemap = NAMEMAP

	lst_out = []
	for m in model_flavors:
		bed_path = bed_path_base.format(m)
		s_args = [script_path, bed_path, annot_path, ctrl_path]
		s_out = subprocess.check_output(s_args)
		parse_output(s_out, lst_out, namemap[m])

	cols_out = [
		"Model", 
		"Minimum Posterior Probability", 
		"Odds Ratio", 
		"-log10 p-Value"
	]

	df_out = pd.DataFrame(lst_out, columns=cols_out)

	return df_out

def plot_enrichment(out_dir, df_out, title, model_flavors, presentation):
	sns.set(style="whitegrid", font="Roboto", rc={'figure.figsize':(8,5)})

	if presentation:
		palette = [COLORMAP_PRES[m] for m in model_flavors]
		names = [NAMEMAP_PRES[m] for m in model_flavors]
	else:
		palette = [COLORMAP[m] for m in model_flavors]
		names = [NAMEMAP[m] for m in model_flavors]

	sns.barplot(
		x="Minimum Posterior Probability", 
		y="Odds Ratio",
		hue="Model",
		data=df_out,
		palette=palette,
		hue_order=names
	)
	if title is not None:
		plt.title(title + "\nOdds Ratios")
	plt.savefig(os.path.join(out_dir, "enrichment_odds.svg"))
	plt.clf()

	sns.barplot(
		x="Minimum Posterior Probability", 
		y="-log10 p-Value",
		hue="Model",
		data=df_out,
		palette=palette,
		hue_order=names
	)
	if title is not None:
		plt.title(title + "\n-log10 p-Values")
	plt.savefig(os.path.join(out_dir, "enrichment_pvals.svg"))
	plt.clf()

def enrichment(bed_path_base, annot_path, script_path, ctrl_path, out_dir, title, model_flavors, presentation=False):
	df_out = run_enrichment(bed_path_base, annot_path, script_path, ctrl_path, model_flavors, presentation)
	data_path = os.path.join(out_dir, "enrichment_data.txt")
	df_out.to_csv(data_path, sep=str("\t"))
	df_out.replace(np.inf, 100, inplace=True)
	plot_enrichment(out_dir, df_out, title, model_flavors, presentation)


if __name__ == '__main__':
	enrichment_path = "/agusevlab/awang/job_data/enrichment"
	script_path = os.path.join(enrichment_path, "pct.sh")

	# Kidney Data
	annot_path = os.path.join(enrichment_path, "KIDNEY_DNASE.E086-DNase.imputed.narrowPeak.bed")
	model_flavors = ["indep", "fmb", "ase", "acav"]

	# Normal
	bed_path_base = "/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_normal_all/ldsr_{0}.bed"
	ctrl_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_normal_all/ctrl.bed"
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_normal_enrichment"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	title = "Kidney RNA-Seq, Open Chromatin Enrichment, Normal Samples"

	enrichment(bed_path_base, annot_path, script_path, ctrl_path, out_dir, title, model_flavors)

	# Tumor
	bed_path_base = "/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_tumor_all/ldsr_{0}.bed"
	ctrl_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_tumor_all/ctrl.bed"
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_enrichment"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	title = "Kidney RNA-Seq, Open Chromatin Enrichment, Tumor Samples"

	enrichment(bed_path_base, annot_path, script_path, ctrl_path, out_dir, title, model_flavors)

	# Tumor, Presentation
	model_flavors_pres = ["indep", "acav", "fmb"]
	bed_path_base = "/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_tumor_all/ldsr_{0}.bed"
	ctrl_path = "/agusevlab/awang/job_data/KIRC_RNASEQ/ldsr_beds/1cv_tumor_all/ctrl.bed"
	out_dir = "/agusevlab/awang/ase_finemap_results/KIRC_RNASEQ/1cv_tumor_enrichment_pres"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	title = None

	enrichment(bed_path_base, annot_path, script_path, ctrl_path, out_dir, title, model_flavors_pres, presentation=True)

	# Prostate Data
	annot_path = os.path.join(enrichment_path, "PRCA_HICHIP.MERGED_Annotated_FDR0.01_ncounts10.E.bed")
	model_flavors = ["indep", "eqtl", "ase", "acav"]

	# Normal
	bed_path_base = "/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_normal_all/ldsr_{0}.bed"
	ctrl_path = "/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_normal_all/ctrl.bed"
	out_dir = "/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_normal_enrichment"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	title = "Prostate ChIP-Seq, Chromatin Looping Enrichment, Normal Samples"

	enrichment(bed_path_base, annot_path, script_path, ctrl_path, out_dir, title, model_flavors)

	# Tumor
	bed_path_base = "/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_tumor_all/ldsr_{0}.bed"
	ctrl_path = "/agusevlab/awang/job_data/prostate_chipseq/ldsr_beds/1cv_tumor_all/ctrl.bed"
	out_dir = "/agusevlab/awang/ase_finemap_results/prostate_chipseq/1cv_tumor_enrichment"
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	title = "Prostate ChIP-Seq, Chromatin Looping Enrichment, Tumor Samples"

	enrichment(bed_path_base, annot_path, script_path, ctrl_path, out_dir, title, model_flavors)
