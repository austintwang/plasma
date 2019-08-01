import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import seaborn as sns
import matplotlib.pyplot as plt

import random
import os
import traceback
import pickle
import signal
from contextlib import contextmanager

if __name__ == '__main__' and __package__ is None:
	__package__ = 'test'
	import sys
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
	sys.path.insert(0, "/agusevlab/awang/plasma")

from . import LocusSimulator

CHROM_LENS = [
	248956422,
	242193529,
	198295559,
	190214555,	
	181538259,
	170805979,
	159345973,
	145138636,
	138394717,
	133797422,
	135086622,
	133275309,
	114364328,
	107043718,
	101991189,			
	90338345,
	83257441,
	80373285,
	58617616,
	64444167,
	46709983,
	50818468,
]

class TimeoutException(Exception): 
	pass

@contextmanager
def time_limit(seconds):
	def signal_handler(signum, frame):
		raise TimeoutException()
	signal.signal(signal.SIGALRM, signal_handler)
	signal.alarm(seconds)
	try:
		yield
	finally:
		signal.alarm(0)

def draw_region(vcf_dir, vcf_name_template):
	regions = [i - 1000000 for i in CHROM_LENS]
	weights = np.array(regions) / np.sum(regions)
	chrom_num = np.random.choice(list(range(1, 23)), p=weights)
	chrom = "chr{0}".format(chrom_num)
	vcf_path = os.path.join(vcf_dir, vcf_name_template.format(chrom))
	start = random.randrange(0, regions[chrom_num-1])

	return chrom, chrom_num, start, vcf_path

def draw_locus(vcf_dir, vcf_name_template, sample_filter, snp_filter, region_size, max_snps, maf_thresh):
	while True:
		try:
			chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)
			with time_limit(100):
				locus = LocusSimulator(
					vcf_path, 
					chrom_num, 
					start, 
					1,
					region_size=region_size,
					max_snps=max_snps,
					sample_filter=sample_filter,
					snp_filter=snp_filter,
					maf_thresh=maf_thresh
				)
		except (ValueError, TimeoutException):
			continue
		if locus.snp_count >= 10:
			break

	return locus

def sim_ld(locus, sample_sizes):
	res_dict = {}
	for s in sample_sizes:
		mult = (s * 2) // locus.num_samples
		rem = (s * 2) % locus.num_samples
		blocks = []
		for _ in range(mult):
			blocks.append(np.arange(locus.num_samples))
		blocks.append(np.random.choice(locus.num_samples, rem, replace=False))
		haps_idx = np.concatenate(blocks)
		haps_sampled = locus.haps[haps_idx]
		np.random.shuffle(haps_sampled)
		hap_A = haps_sampled[:s]
		hap_B = haps_sampled[s:]

		haps_mean = np.mean(haps_sampled, axis=0)
		haps_centered = haps_sampled - haps_mean
		haps_cov = np.nan_to_num(np.cov(haps_centered.T))
		haps_var = np.diagonal(haps_cov)
		haps_corr = haps_cov / np.sqrt(np.outer(haps_var, haps_var))
		haps_corr = np.nan_to_num(haps_corr)
		np.fill_diagonal(haps_corr, 1.0)
		# print(haps_sampled) ####
		# print(haps_corr) ####

		dosage = hap_A + hap_B
		dosage_means = np.mean(dosage, axis=0)
		dosage_centered = dosage - dosage_means
		dosage_cov = np.nan_to_num(np.cov(dosage_centered.T))
		dosage_var = np.diagonal(dosage_cov)
		dosage_corr = dosage_cov / np.sqrt(np.outer(dosage_var, dosage_var))
		dosage_corr = np.nan_to_num(dosage_corr)
		np.fill_diagonal(dosage_corr, 1.0)

		phases = hap_A - hap_B
		phases_cov = np.nan_to_num(np.cov(phases.T))
		phases_var = np.diagonal(phases_cov)
		phases_corr = phases_cov / np.sqrt(np.outer(phases_var, phases_var))
		phases_corr = np.nan_to_num(phases_corr)
		np.fill_diagonal(phases_corr, 1.0)

		res_dict[s] = {"haps": haps_corr, "dosage": dosage_corr, "phases": phases_corr} 

	return res_dict

def plot_scatter(x, y, x_lab, y_lab, title, fname, output_dir):
	sns.scatterplot(x=x, y=y)
	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	plt.title(title)
	plt.savefig(os.path.join(output_dir, fname))
	plt.clf()

def plot_lds(res_dict, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	sns.set(style="whitegrid", font="Roboto")

	for k, v in res_dict.items():
		h = v["haps"]
		f_h = h.flatten()
		e_h = np.linalg.eigvals(h)
		l_h = "Haplotype LD"

		d = v["dosage"]
		f_d = d.flatten()
		e_d = np.linalg.eigvals(d)
		l_d = "Dosage LD"

		p = v["haps"]
		f_p = p.flatten()
		e_p = np.linalg.eigvals(p)
		l_p = "Phasing LD"

		f_tmp = "LD Correlations\n{0} LD vs. {1} LD, {2} Samples"
		e_tmp = "LD Eigenvalues\n{0} LD vs. {1} LD, {2} Samples"

		f_f = "corr_{0}_{1}_{2}.svg"
		e_f = "eigs_{0}_{1}_{2}.svg"

		plot_scatter(f_h, f_d, l_h, l_d, f_tmp.format(l_h, l_d, k), f_f.format("h", "d", k), output_dir)
		plot_scatter(e_h, e_d, l_h, l_d, e_tmp.format(l_h, l_d, k), e_f.format("h", "d", k), output_dir)

		plot_scatter(f_h, f_p, l_h, l_p, f_tmp.format(l_h, l_p, k), f_f.format("h", "p", k), output_dir)
		plot_scatter(e_h, e_p, l_h, l_p, e_tmp.format(l_h, l_p, k), e_f.format("h", "p", k), output_dir)

		plot_scatter(f_d, f_p, l_d, l_p, f_tmp.format(l_d, l_p, k), f_f.format("d", "p", k), output_dir)
		plot_scatter(e_d, e_p, l_d, l_p, e_tmp.format(l_d, l_p, k), f_f.format("d", "p", k), output_dir)


if __name__ == '__main__':
	vcf_dir = "/agusevlab/awang/job_data/sim_coloc/vcfs/"
	vcf_name_template = "ALL.{0}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
	sample_filter_path = "/agusevlab/awang/job_data/sim_coloc/vcfs/integrated_call_samples_v3.20130502.ALL.panel"
	snp_filter_path = "/agusevlab/awang/job_data/sim_coloc/1000g/snp_filter.pickle"

	output_dir = "/agusevlab/awang/ase_finemap_results/Simulations/ld/"

	region_size = None
	max_snps = 100
	maf_thresh = .01

	sample_sizes = [1000, 500, 200, 100, 50, 10]

	sample_filter_data = pd.read_csv(
		sample_filter_path, 
		sep="\t", 
		usecols=["sample", "super_pop"]
	)
	
	sample_filter = set(
		sample_filter_data.loc[
			sample_filter_data["super_pop"]=="EUR",
			["sample"]
		].to_numpy().flatten()
	)

	with open(snp_filter_path, "rb") as snp_filter_file:
		snp_filter = pickle.load(snp_filter_file)

	locus = draw_locus(vcf_dir, vcf_name_template, sample_filter, snp_filter, region_size, max_snps, maf_thresh)
	res_dict = sim_ld(locus, sample_sizes)
	plot_lds(res_dict, output_dir)




