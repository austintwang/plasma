#!/usr/bin/env python3

import os
import random
import traceback
import pickle
import signal
from contextlib import contextmanager
# import gc

import numpy as np
import pandas as pd

if __name__ == '__main__' and __package__ is None:
	__package__ = 'test'
	import sys
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
	sys.path.insert(0, "/agusevlab/awang/plasma")

from . import Finemap, LocusSimulator, Caviar, CaviarASE, FmBenner, Rasqual

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

class DummyFinemap(Finemap):
	def __init__(self, ppas, causal_set):
		self.ppas = ppas
		self.causal_set = causal_set

	def get_ppas(self):
		return self.ppas

	def get_causal_set(self, confidence):
		return self.causal_set

def draw_region(vcf_dir, vcf_name_template):
	regions = [i - 1000000 for i in CHROM_LENS]
	weights = np.array(regions) / np.sum(regions)
	chrom_num = np.random.choice(list(range(1, 23)), p=weights)
	chrom = "chr{0}".format(chrom_num)
	vcf_path = os.path.join(vcf_dir, vcf_name_template.format(chrom))
	start = random.randrange(0, regions[chrom_num-1])

	return chrom, chrom_num, start, vcf_path

def sim_random(vcf_dir, vcf_name_template, sample_filter, snp_filter, params):
	while True:
		try:
			chrom, chrom_num, start, vcf_path = draw_region(vcf_dir, vcf_name_template)
			with time_limit(100):
				locus = LocusSimulator(
					vcf_path, 
					chrom_num, 
					start, 
					params["num_causal"],
					region_size=params["region_size"],
					max_snps=params["max_snps"],
					sample_filter=sample_filter,
					snp_filter=snp_filter,
					maf_thresh=params["maf_thresh"]
				)
		except (ValueError, TimeoutException):
			continue
		if locus.snp_count >= 10:
			break

	qtl_data = locus.sim_asqtl(
		params["num_samples"],
		params["coverage"],
		params["std_al_dev"],
		params["herit_qtl"],
		params["herit_as"],
		params["overdispersion"],
		switch_error=params.get("switch_error", 0.),
		blip_error=params.get("blip_error", 0.)
	)

	return locus, qtl_data

def run_model(model_cls, inputs, model_name, model_updates):
	if inputs["num_samples"] == "Perfect":
		causal_set = inputs["causal_config"]
		ppas = inputs["causal_config"]
		model = DummyFinemap(ppas, causal_set)
	else:
		inputs_model = inputs.copy()
		inputs_model.update(model_updates)
		param_renames = {
			"total_exp_herit_prior": inputs_model["herit_qtl"],
			"imbalance_herit_prior": inputs_model["herit_as"],
			"num_ppl": inputs_model["num_samples"],
		}
		inputs_model.update(param_renames)

		model = model_cls(**inputs_model)
		model.initialize()

		if inputs_model["search_mode"] == "exhaustive":
			model.search_exhaustive(inputs["min_causal"], inputs["max_causal"])
		elif inputs_model["search_mode"] == "shotgun":
			model.search_shotgun(
				inputs_model["min_causal"], 
				inputs_model["max_causal"], 
				inputs_model["prob_threshold"], 
				inputs_model["streak_threshold"], 
				inputs_model["search_iterations"]
			)
		causal_config = inputs_model["causal_config"]

		causal_set = model.get_causal_set(inputs_model["confidence"])
		ppas = model.get_ppas()
		recall = 1
		for i in np.nonzero(causal_config)[0]:
			if causal_set[i] != 1:
				recall = 0
		selections = np.flip(np.argsort(ppas))
		causals = causal_config[selections]
		inclusion = np.cumsum(causals) / np.sum(causal_config)

	result = {
		"causal_set": causal_set,
		"ppas": ppas,
		"recall": recall,
		"inclusion": inclusion
	}

	result.update(inputs)
	result["model"] = model_name
	result["complete"] = True
	return result

def sim_test(
		out_dir, 
		batch_size, 
		batch_num, 
		params_path
	):
	with open(params_path, "rb") as params_file:
		params = pickle.load(params_file)
	test_type = params["test_type"]
	vcf_dir = params["vcf_dir"] 
	vcf_name_template = params["vcf_name_template"]
	model_flavors = params["model_flavors"]

	sample_filter_data = pd.read_csv(
		params["sample_filter_path"], 
		sep="\t", 
		usecols=["sample", "super_pop"]
	)
	
	sample_filter = set(
		sample_filter_data.loc[
			sample_filter_data["super_pop"]=="EUR",
			["sample"]
		].to_numpy().flatten()
	)

	with open(params["snp_filter_path"], "rb") as snp_filter_file:
		snp_filter = pickle.load(snp_filter_file)

	output = []
	for _ in range(batch_size):
		try:
			locus, qtl_data = sim_random(
				vcf_dir, 
				vcf_name_template, 
				sample_filter,
				snp_filter,
				params
			)
			# print(qtl_data) ####
			# print(qtl_data["counts_A"]) ####
			# print(qtl_data["counts_B"]) ####
			sim_data = {
				"sim_type": test_type,
				"chrom": locus.chrom,
				"locus_pos": locus.start,
				"locus_size": locus.region_size,
				"snp_ids": locus.snp_ids,
				"num_snps": locus.snp_count,
				"causal_config": locus.causal_config,
			}
			sim_data.update(qtl_data)

			inputs = params.copy()
			inputs.update(sim_data)
			# print(inputs["hap_A"].shape()) ####

		except Exception as e:
			raise ####
			trace = traceback.format_exc()
			message = repr(e)
			result = {"complete": False, "error": message, "traceback": trace}
			if "full" in model_flavors:
				result_full = result.copy()
				result_full.update({"model": "full"})
				output.append(result_full)
			if "indep" in model_flavors:
				result_indep = result.copy()
				result_indep.update({"model": "indep"})
				output.append(result_indep)
			if "ase" in model_flavors:
				result_ase = result.copy()
				result_ase.update({"model": "ase"})
				output.append(result_ase)
			if "acav" in model_flavors:
				result_ecav = result.copy()
				result_ecav.update({"model": "acav"})
				output.append(result_ecav)
			if "eqtl" in model_flavors:
				result_eqtl = result.copy()
				result_eqtl.update({"model": "eqtl"})
			if "cav" in model_flavors:
				result_eqtl = result.copy()
				result_eqtl.update({"model": "cav"})
				output.append(result_eqtl)
			if "rasq" in model_flavors:
				result_eqtl = result.copy()
				result_eqtl.update({"model": "rasq"})
				output.append(result_eqtl)
			if "bfm" in model_flavors:
				result_eqtl = result.copy()
				result_eqtl.update({"model": "bfm"})
				output.append(result_eqtl)

			continue

		if "full" in model_flavors:
			try:
				param_updates = {}
				result_full = run_model(Finemap, inputs, "full", param_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_full = {
					"model": "full",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_full)

		if "indep" in model_flavors:
			try:
				param_updates = {"cross_corr_prior": 0.}
				result_indep = run_model(Finemap, inputs, "indep", param_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_indep = {
					"model": "indep",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_indep)
			
		if "ase" in model_flavors:
			try:
				param_updates = {"as_only": True}
				result_ase = run_model(Finemap, inputs, "ase", param_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_ase = {
					"model": "ase",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_ase)
			
		if "acav" in model_flavors:
			try:
				param_updates = {}
				result_acav = run_model(CaviarASE, inputs, "acav", param_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_acav = {
					"model": "acav",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_ecav)
			
		if "eqtl" in model_flavors:
			try:
				param_updates = {"qtl_only": True}
				result_eqtl = run_model(Finemap, inputs, "eqtl", param_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_eqtl = {
					"model": "eqtl",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_eqtl)

		if "cav" in model_flavors:
			try:
				param_updates = {"qtl_only": True}
				result_ecav = run_model(Caviar, inputs, "cav", param_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_cav = {
					"model": "cav",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_cav)

		if "rasq" in model_flavors:
			try:
				param_updates = {"as_only": True}
				# print("test0") ####
				result_rasq = run_model(Rasqual, inputs, "rasq", param_updates)
				print(result_rasq) ####
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_rasq = {
					"model": "rasq",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_rasq)

		if "fmb" in model_flavors:
			try:
				param_updates = {"qtl_only": True}
				result_fmb = run_model(FmBenner, inputs, "fmb", param_updates)
			except Exception as e:
				raise ####
				trace = traceback.format_exc()
				message = repr(e)
				result_fmb = {
					"model": "fmb",
					"complete": False, 
					"error": message, 
					"traceback": trace
				}
			finally:
				output.append(result_fmb)
			
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	output_name = "{0}_out_{1}.pickle".format(params["test_name"], batch_num)
	output_return = os.path.join(out_dir, output_name)
	with open(output_return, "wb") as output_file:
		pickle.dump(output, output_file)

if __name__ == '__main__':
	out_dir = sys.argv[1]
	batch_size = int(sys.argv[2])
	batch_num = int(sys.argv[3])
	params_path = sys.argv[4]
	sim_test(
		out_dir, 
		batch_size, 
		batch_num, 
		params_path
	)