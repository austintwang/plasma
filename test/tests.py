from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import scipy.misc as spm

from . import Benchmark, Benchmark2d
from . import Haplotypes

def dummy_test():
	params = {
		"num_snps": 200,
		"num_ppl": 95,
		"overdispersion": 0.05,
		"prop_noise_eqtl": 0.95,
		"prop_noise_ase": 0.50,
		"std_fraction": 0.75,
		"min_causal": 1,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"primary_var": "std_fraction",
		"primary_var_display": "Standard Allelic Fraction",
		"test_count": 1,
		"test_name": "dummy_test",
		"iterations": 1,
		"confidence": 0.95 
	}
	tests = [0.75,]
	bm = Benchmark(params)
	for t in tests:
		bm.test(std_fraction=t)
	bm.output_summary()

def dummy_test_2d():
	params = {
		"num_snps": 90,
		"num_ppl": 95,
		"overdispersion": 0.05,
		"prop_noise_eqtl": 0.95,
		"prop_noise_ase": 0.6,
		"std_fraction": None,
		"num_causal": 1,
		"coverage": None,
		"search_mode": "exhaustive",
		"min_causal": 1,
		"max_causal": 2,
		"primary_var": "std_fraction",
		"primary_var_display": "Standard Allelic Fraction",
		"secondary_var": "coverage",
		"secondary_var_display": "Coverage",
		"test_count": 4,
		"test_count_primary": 2,
		"test_count_secondary": 2,
		"test_name": "dummy_test_2d",
		"iterations": 50,
		"confidence": 0.95
	}
	
	ptests = [0.6, 0.8 ]
	stests = [10, 100]
	
	bm = Benchmark2d(params)
	for s in stests:
		for p in ptests:
			bm.test(std_fraction=p, coverage=s)
	bm.output_summary()

def confidence_test():
	params = {
		"num_snps": 00,
		"num_ppl": 95,
		"overdispersion": 0.05,
		"prop_noise_eqtl": 0.95,
		"prop_noise_ase": 0.50,
		"std_fraction": 0.75,
		"min_causal": 1,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"primary_var": "confidence",
		"primary_var_display": "Confidence of Causal Set",
		"test_count": 9,
		"test_name": "confidence_test",
		"iterations": 100,
		"confidence": None 
	}
	tests = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	bm = Benchmark(params)
	for t in tests:
		bm.test(confidence=t)
	bm.output_summary()

def imbalance():
	params = {
		"num_snps": 100,
		"num_ppl": 95,
		"overdispersion": 0.05,
		"herit_eqtl": 0.05,
		"herit_ase": 0.4,
		"std_fraction": None,
		"min_causal": 1,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"primary_var": "std_fraction",
		"primary_var_display": "Imbalance (Standard Allelic Fraction)",
		"test_count": 2,
		"test_name": "imbalance",
		"test_path": "/home/austin/Documents/Gusev/Results/ase_finemap_results/Simulations",
		"iterations": 500,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase", "cav", "acav"]) 
	}
	tests = [0.6, 0.8]
	bm = Benchmark(params)
	for t in tests:
		bm.test(std_fraction=t)
	bm.output_summary()

def fraction_vs_coverage():
	params = {
		"num_snps": 100,
		"num_ppl": 95,
		"overdispersion": 0.05,
		"herit_eqtl": 0.05,
		"herit_ase": 0.4,
		"std_fraction": None,
		"num_causal": 1,
		"coverage": None,
		"search_mode": "exhaustive",
		"min_causal": 1,
		"max_causal": 1,
		"primary_var": "std_fraction",
		"primary_var_display": "Imbalance (Standard Allelic Fraction)",
		"secondary_var": "coverage",
		"secondary_var_display": "Coverage",
		"test_count": 54,
		"test_count_primary": 9,
		"test_count_secondary": 6,
		"test_name": "fraction_vs_coverage",
		"test_path": "/home/austin/Documents/Gusev/Results/ase_finemap_results/Simulations",
		"iterations": 500,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase", "acav"])
	}
	
	ptests = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
	stests = [10, 20, 50, 100, 500, 1000]
	
	bm = Benchmark2d(params)
	for s in stests:
		for p in ptests:
			bm.test(std_fraction=p, coverage=s)
	bm.output_summary()

def fraction_vs_noise():
	params = {
		"num_snps": 100,
		"num_ppl": 95,
		"overdispersion": 0.05,
		"herit_eqtl": 0.05,
		"herit_ase": None,
		"std_fraction": None,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"min_causal": 1,
		"max_causal": 1,
		"primary_var": "std_fraction",
		"primary_var_display": "Imbalance (Standard Allelic Fraction)",
		"secondary_var": "herit_ase",
		"secondary_var_display": "ASE Heritability",
		"test_count": 54,
		"test_count_primary": 9,
		"test_count_secondary": 6,
		"test_name": "fraction_vs_ase_noise",
		"test_path": "/home/austin/Documents/Gusev/Results/ase_finemap_results/Simulations",
		"iterations": 500,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase", "acav"])
	}
	
	ptests = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95 ]
	stests = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	
	bm = Benchmark2d(params)
	for s in stests:
		for p in ptests:
			bm.test(std_fraction=p, herit_ase=s)
	bm.output_summary()

def multi_cv():
	params = {
		"num_snps": 100,
		"num_ppl": 95,
		"overdispersion": 0.05,
		"herit_eqtl": 0.05,
		"herit_ase": 0.6,
		"std_fraction": 0.65,
		"min_causal": 1,
		"coverage": 100,
		"search_mode": "shotgun",
		"prob_threshold": 0.001,
		"streak_threshold": 1000,
		"search_iterations": None, 
		"max_causal": None,
		"num_causal": None,
		"primary_var": "num_causal",
		"primary_var_display": "Number of Causal Variants",
		"test_count": 3,
		"test_name": "multi_cv",
		"test_path": "/home/austin/Documents/Gusev/Results/ase_finemap_results/Simulations",
		"iterations": 500,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase"]) 
	}
	tests = [1, 2, 3]
	bm = Benchmark(params)
	for t in tests:
		it = int(spm.comb(params["num_snps"], t))
		bm.test(max_causal=t, num_causal=t, search_iterations=it)
	bm.output_summary()

if __name__ == "__main__":
	# dummy_test()
	# dummy_test_2d()
	# confidence_test()
	imbalance()
	# fraction_vs_coverage()
	# fraction_vs_noise()
	# multi_cv()
