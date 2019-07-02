import subprocess
import os
import pickle

class Dispatcher(object):
	def __init__(self, script_path, batch_size):
		self.script_path = script_path
		self.batch_size = batch_size
		self.jobs = []

	def add_job(self, out_dir, params_path, params, num_tasks):
		with open(params_path, "wb") as params_file:
			pickle.dump(params, params_file)

		job_name = params["test_name"]

		num_jobs = -(-num_tasks // self.batch_size)
		batches = [self.batch_size for i in range(num_jobs)]
		batches[-1] -= -num_tasks % self.batch_size

		for ind, val in enumerate(batches):
			job_args = [
				"sbatch", 
				"-J", 
				job_name, 
				self.script_path,
				str(val),
				str(ind),
				params_path
			]
			self.jobs.append(job_args)

		raise Exception ####

	def submit(self):
		for i in jobs:
			subprocess.call(i)

def test_shared_causal(
	disp, 
	data_info,
	params_dir, 
	out_dir_base, 
	qtl_sizes, 
	gwas_sizes, 
	gwas_herits, 
	num_trials,
	script_path,
):
	params_base = {
		"test_type": "shared",
		"region_size": 1000000,
		"num_samples_qtl": None,
		"num_samples_gwas": None,
		"maf_thresh": 0.1,
		"overdispersion": 0.05,
		"herit_eqtl": 0.05,
		"herit_ase": 0.4,
		"herit_gwas": None,
		"std_al_dev": 0.7,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"min_causal": 1,
		"max_causal": 1,
		"test_name": None,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase", "ecav"])
	}
	out_dir = os.path.join(out_dir_base, "shared")

	for i in qtl_sizes:
		for j in gwas_sizes:
			for k in gwas_herits:
				test_name = "q_{0}_g_{1}_h_{2}_shared".format(i, j, k)
				param_updates = {
					"test_name": test_name,
					"num_samples_qtl": i,
					"num_samples_gwas": j,
					"herit_gwas": k,
				}
				params = params_base.copy()
				params.update(param_updates)
				params.update(data_info)
				params_path = os.path.join(params_dir, test_name + ".pickle")
				disp.add_job(out_dir, params_path, params, num_trials)

def test_unshared_corr(
	disp, 
	data_info,
	params_dir, 
	out_dir_base, 
	qtl_sizes, 
	gwas_sizes, 
	ld_thresh, 
	num_trials,
	script_path,
):
	params_base = {
		"test_type": "corr",
		"region_size": 1000000,
		"num_samples_qtl": None,
		"num_samples_gwas": None,
		"maf_thresh": 0.1,
		"overdispersion": 0.05,
		"herit_eqtl": 0.05,
		"herit_ase": 0.4,
		"herit_gwas": .01/100,
		"corr_thresh": None,
		"std_al_dev": 0.7,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"min_causal": 1,
		"max_causal": 1,
		"test_name": None,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase", "ecav"])
	}
	out_dir = os.path.join(out_dir_base, "corr")

	for i in qtl_sizes:
		for j in gwas_sizes:
			for k in ld_thresh:
				test_name = "q_{0}_g_{1}_h_{2}_corr".format(i, j, k)
				param_updates = {
					"test_name": test_name,
					"num_samples_qtl": i,
					"num_samples_gwas": j,
					"corr_thresh": k,
				}
				params = params_base.copy()
				params.update(param_updates)
				params.update(data_info)
				params_path = os.path.join(params_dir, test_name + ".pickle")
				disp.add_job(out_dir, params_path, params, num_trials)

if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))

	script_path = os.path.join(curr_path, "coloc_test.py")
	batch_size = 50
	num_trials = 500

	disp = Dispatcher(script_path, batch_size)

	data_info = {
		"vcf_dir": "/agusevlab/awang/job_data/sim_coloc/vcfs/",
		"vcf_name_template": "ALL.{0}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz",
		"sample_filter_path": "/agusevlab/awang/job_data/sim_coloc/vcfs/integrated_call_samples_v3.20130502.ALL.panel",
		"snp_filter_path": "/agusevlab/awang/job_data/sim_coloc/1000g/snp_filter.pickle"
	}
	params_dir = "/agusevlab/awang/job_data/sim_coloc/params/"
	if not os.path.exists(params_dir):
		os.makedirs(params_dir)
	out_dir_base = "/agusevlab/awang/job_data/sim_coloc/outs/"
	if not os.path.exists(out_dir_base):
		os.makedirs(out_dir_base)

	qtl_sizes = [10, 50, 100, 200, 500]
	gwas_sizes = [10000, 50000, 100000, 200000, 500000]

	qtl_sizes = [500, 200, 100, 50, 10]
	gwas_sizes = [500000, 200000, 100000, 50000, 10000]

	gwas_herits = [.01/100, .05/1000]
	test_shared_causal(
		disp, 
		data_info,
		params_dir, 
		out_dir_base, 
		qtl_sizes, 
		gwas_sizes, 
		gwas_herits, 
		num_trials,
		script_path,
	)

	ld_thresh = [0., 0.2, 0.4, 0.8]
	test_unshared_corr(
		disp, 
		data_info,
		params_dir, 
		out_dir_base, 
		qtl_sizes, 
		gwas_sizes, 
		ld_thresh, 
		num_trials,
		script_path,
	)

	disp.submit()