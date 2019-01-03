from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

try:
	import cPickle as pickle
except ImportError:
	import pickle

def plot_dist(result, out_dir, name):
	set_sizes_full = result["set_sizes_full"]
	set_sizes_indep = result["set_sizes_indep"]
	set_sizes_eqtl = result["set_sizes_eqtl"]
	set_sizes_ase = result["set_sizes_ase"]
	set_sizes_caviar_ase = result["set_sizes_caviar_ase"]

	try:
		sns.set(style="white")
		sns.distplot(
			set_sizes_full,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="Full"
		)
		sns.distplot(
			set_sizes_indep,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="Independent Likelihoods"
		)
		sns.distplot(
			set_sizes_eqtl,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="eQTL-Only"
		)
		sns.distplot(
			set_sizes_ase,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="ASE-Only"
		)
		sns.distplot(
			set_sizes_caviar_ase,
			hist=False,
			kde=True,
			kde_kws={"linewidth": 3, "shade":True},
			label="CAVIAR-ASE"
		)
		plt.xlim(0, None)
		plt.legend(title="Model")
		plt.xlabel("Set Size")
		plt.ylabel("Density")
		plt.title("Distribution of Causal Set Sizes: {0}".format(name))
		plt.savefig(os.path.join(out_dir, "set_size_distribution.svg"))
		plt.clf()
	except Exception:
		plt.clf()

def interpret(target_dir, out_dir, name):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	targets = os.listdir(target_dir)

	summary = {
		"set_sizes_full": [],
		"set_sizes_indep": [],
		"set_sizes_eqtl": [],
		"set_sizes_ase": [],
		"set_sizes_caviar_ase": []
	}
	failed_jobs = []

	for t in targets:
		result_path = os.path.join(target_dir, t, "output.pickle")
		if os.path.isfile(result_path):
			with open(result_path, "rb") as result_file:
				result = pickle.load(result_file)
		else:
			failed_jobs.append(t)
			continue

		summary["set_sizes_full"].append(np.size(result["causal_set_full"])) 
		summary["set_sizes_indep"].append(np.size(result["causal_set_indep"]))
		summary["set_sizes_eqtl"].append(np.size(result["causal_set_eqtl"])) 
		summary["set_sizes_ase"].append(np.size(result["causal_set_ase"])) 
		summary["set_sizes_caviar_ase"].append(np.size(result["causal_set_caviar_ase"])) 

	with open(os.path.join(out_dir, "failed_jobs.txt"), "w") as fail_out:
		fail_out.write("\n".join(failed_jobs))

	plot_dist(summary, out_dir, name)

	# return summary

def run(out_dir, margin, hyperparams, num_tasks, poll_freq, script_path, name):
	return interpret(complete_pool, out_dir, jobs_dir, name)

if __name__ == '__main__':
	# Kidney 1CV All
	target_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ_ASVCF/outs/1cv_all"
	out_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ_ASVCF/results/1cv_all"
	name = "Kidney RNA-Seq\nAll Individiuals, 1 Causal Variant"

	interpret(target_dir, out_dir, name)