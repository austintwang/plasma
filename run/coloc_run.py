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
	__package__ = 'run'
	import sys
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
	sys.path.insert(0, "/agusevlab/awang/plasma")

def run_model(inputs, model_name, model_qtl_updates):
	inputs_qtl = inputs.copy()
	qtl_updates = {
		"total_exp_herit_prior": inputs_qtl["herit_qtl"],
		"imbalance_herit_prior": inputs_qtl["herit_as"],
		"num_ppl": inputs_qtl["num_samples_qtl"],
	}
	inputs_qtl.update(qtl_updates)
	inputs_qtl.update(model_qtl_updates)

	model_qtl = Finemap(**inputs_qtl)
	model_qtl.initialize()

	if inputs_qtl["search_mode"] == "exhaustive":
		model_qtl.search_exhaustive(inputs["min_causal"], inputs["max_causal"])
	elif inputs_qtl["search_mode"] == "shotgun":
		model_qtl.search_shotgun(
			inputs_qtl["min_causal"], 
			inputs_qtl["max_causal"], 
			inputs_qtl["prob_threshold"], 
			inputs_qtl["streak_threshold"], 
			inputs_qtl["search_iterations"]
		)

	causal_set_qtl = model_qtl.get_causal_set(inputs_qtl["confidence"])
	ppas_qtl = model_qtl.get_ppas()

	inputs_gwas = inputs.copy()
	gwas_updates = {
		"total_exp_herit_prior": inputs_gwas["herit_gwas"],
		"total_exp_stats": inputs_gwas["z_gwas"],
		"total_exp_corr": inputs_gwas["ld_gwas"],
		"num_ppl": inputs_gwas["num_samples_gwas"],
		"qtl_only": True
	}
	inputs_gwas.update(gwas_updates)

	model_gwas = Finemap(**inputs_gwas)
	model_gwas.initialize()

	if inputs_gwas["search_mode"] == "exhaustive":
		model_gwas.search_exhaustive(inputs["min_causal"], inputs["max_causal"])
	elif inputs_gwas["search_mode"] == "shotgun":
		model_gwas.search_shotgun(
			inputs_gwas["min_causal"], 
			inputs_gwas["max_causal"], 
			inputs_gwas["prob_threshold"], 
			inputs_gwas["streak_threshold"], 
			inputs_gwas["search_iterations"]
		)

	causal_set_gwas = model_gwas.get_causal_set(inputs_qtl["confidence"])
	ppas_gwas = model_gwas.get_ppas()

	clpps = model_qtl.coloc_clpps(model_gwas)
	h0, h1, h2, h3, h4 = model_qtl.coloc_hyps(model_gwas)

	result = {
		"causal_set_qtl": causal_set_qtl,
		"causal_set_gwas": causal_set_gwas,
		"ppas_qtl": ppas_qtl,
		"ppas_gwas": ppas_gwas,
		"clpps": clpps,
		"h0": h0,
		"h1": h1,
		"h2": h2,
		"h3": h3,
		"h4": h4
	}

	result.update(inputs)
	result["model"] = model_name
	result["complete"] = True
	return result

def run_ecav(inputs, model_name, model_qtl_updates):
	inputs_qtl = inputs.copy()
	qtl_updates = {
		"total_exp_herit_prior": inputs_qtl["herit_qtl"],
		"imbalance_herit_prior": inputs_qtl["herit_as"],
		"num_ppl": inputs_qtl["num_samples_qtl"],
	}
	inputs_qtl.update(qtl_updates)
	inputs_qtl.update(model_qtl_updates)

	model_qtl = Finemap(**inputs_qtl)
	model_qtl.initialize()

	inputs_gwas = inputs.copy()
	gwas_updates = {
		"total_exp_herit_prior": inputs_gwas["herit_gwas"],
		"total_exp_stats": inputs_gwas["z_gwas"],
		"total_exp_corr": inputs_gwas["ld_gwas"],
		"num_ppl": inputs_gwas["num_samples_gwas"],
		"qtl_only": True
	}
	inputs_gwas.update(gwas_updates)

	model_gwas = Finemap(**inputs_gwas)
	model_gwas.initialize()

	model_ecaviar = EvalECaviar(
		model_qtl,
		model_gwas, 
		inputs["confidence"], 
		inputs["max_causal"]
	)
	model_ecaviar.run()

	causal_set_qtl = model_ecaviar.causal_set_qtl
	causal_set_gwas = model_ecaviar.causal_set_gwas
	ppas_qtl = model_ecaviar.post_probs_qtl
	ppas_gwas = model_ecaviar.post_probs_gwas
	clpps = model_ecaviar.clpp
	h4 = model_ecaviar.h4

	result = {
		"causal_set_qtl": causal_set_qtl,
		"causal_set_gwas": causal_set_gwas,
		"ppas_qtl": ppas_qtl,
		"ppas_gwas": ppas_gwas,
		"clpps": clpps,
		"h4": h4
	}

	result.update(inputs)
	result["model"] = model_name
	result["complete"] = True
	return result