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
			err_name = os.path.join(out_dir, "{0}_{1}.err".format(job_name, ind))
			job_args = [
				"sbatch", 
				"-J", 
				job_name, 
				"-e",
				err_name,
				self.script_path,
				out_dir,
				str(val),
				str(ind),
				params_path,	
			]
			self.jobs.append(job_args)

	def submit(self):
		timeout = "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"
		for i in self.jobs:
			print(" ".join(i)) ####
			raise Exception ####
			while True:
				try:
					submission = subprocess.run(i, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
					print(str(submission.stdout, 'utf-8').rstrip())
					break
				except subprocess.CalledProcessError as e:
					# print(e.stdout) ####
					err = str(e.stderr, 'utf-8').rstrip()
					print(err)
					if err == timeout:
						print("Retrying Submit")
						continue
					else:
						raise e

def test_dev_cov(
	disp, 
	data_info,
	params_dir, 
	out_dir_base, 
	std_al_dev,
	coverage, 
	num_trials,
	script_path
):
	params_base = {
		"test_type": "dev_cov",
		"region_size": None,
		"max_snps": 100,
		"num_samples": 100,
		"maf_thresh": 0.01,
		"overdispersion": 0.05,
		"herit_qtl": 0.05,
		"herit_as": 0.4,
		"std_al_dev": None,
		"num_causal": 1,
		"coverage": None,
		"search_mode": "exhaustive",
		"prob_threshold": 0.001,
		"streak_threshold": 1000,
		"search_iterations": None, 
		"min_causal": 1,
		"max_causal": 1,
		"test_name": None,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase", "acav", "rasq", "bfm"])
	}
	params_base.update(data_info)

	out_dir = os.path.join(out_dir_base, params_base["test_type"])
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	for i in std_al_dev:
		for j in coverage:
			test_name = "s_{0}_c_{1}".format(i, j)
			param_updates = {
				"test_name": test_name,
				"std_al_dev": i,
				"coverage": j,
			}
			params = params_base.copy()
			params.update(param_updates)
			params.update(data_info)
			params_path = os.path.join(params_dir, test_name + ".pickle")
			disp.add_job(out_dir, params_path, params, num_trials)

def test_dev_herit(
	disp, 
	data_info,
	params_dir, 
	out_dir_base, 
	std_al_dev,
	herit_as, 
	num_trials,
	script_path
):
	params_base = {
		"test_type": "dev_herit",
		"region_size": None,
		"max_snps": 100,
		"num_samples": 100,
		"maf_thresh": 0.01,
		"overdispersion": 0.05,
		"herit_qtl": 0.05,
		"herit_as": None,
		"std_al_dev": None,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"prob_threshold": 0.001,
		"streak_threshold": 1000,
		"search_iterations": None, 
		"min_causal": 1,
		"max_causal": 1,
		"test_name": None,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase", "acav", "rasq", "bfm"])
	}
	params_base.update(data_info)

	out_dir = os.path.join(out_dir_base, params_base["test_type"])
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	for i in std_al_dev:
		for j in herit_as:
			test_name = "s_{0}_h_{1}".format(i, j)
			param_updates = {
				"test_name": test_name,
				"std_al_dev": i,
				"herit_as": j,
			}
			params = params_base.copy()
			params.update(param_updates)
			params.update(data_info)
			params_path = os.path.join(params_dir, test_name + ".pickle")
			disp.add_job(out_dir, params_path, params, num_trials)

def test_multi_cv(
	disp, 
	data_info,
	params_dir, 
	out_dir_base, 
	causal_vars, 
	num_trials,
	script_path,
):
	params_base = {
		"test_type": "multi_cv",
		"region_size": None,
		"max_snps": 100,
		"num_samples": 100,
		"maf_thresh": 0.01,
		"overdispersion": 0.05,
		"herit_qtl": 0.05,
		"herit_as": 0.4,
		"herit_as_man": 0.1,
		"std_al_dev": 0.7,
		"num_causal": None,
		"coverage": 100,
		"search_mode": "shotgun",
		"prob_threshold": 0.001,
		"streak_threshold": 1000,
		"search_iterations": None, 
		"min_causal": 1,
		"max_causal": None,
		"test_name": None,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase", "acav", "bfm"])
	}
	params_base.update(data_info)

	out_dir = os.path.join(out_dir_base, params_base["test_type"])
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	for i in causal_vars:
		test_name = "k_{0}".format(i)
		it = int(spm.comb(params_base["max_snps"], i))
		param_updates = {
			"test_name": test_name,
			"max_causal": i + 1,
			"num_causal": i,
			"search_iterations": it,
		}
		params = params_base.copy()
		params.update(param_updates)
		params.update(data_info)
		params_path = os.path.join(params_dir, test_name + ".pickle")
		disp.add_job(out_dir, params_path, params, num_trials)

def test_imperfect_phs(
	disp, 
	data_info,
	params_dir, 
	out_dir_base, 
	phs_errors, 
	num_trials,
	script_path
):
	params_base = {
		"test_type": "imperfect_phs",
		"region_size": None,
		"max_snps": 100,
		"num_samples": 100,
		"maf_thresh": 0.01,
		"overdispersion": 0.05,
		"herit_qtl": 0.05,
		"herit_as": 0.4,
		"std_al_dev": 0.7,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"prob_threshold": 0.001,
		"streak_threshold": 1000,
		"search_iterations": None, 
		"min_causal": 1,
		"max_causal": 1,
		"test_name": None,
		"confidence": 0.95,
		"model_flavors": set(["indep", "ase", "rasq"])
	}
	params_base.update(data_info)

	out_dir = os.path.join(out_dir_base, params_base["test_type"])
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	for s, b in phs_errors:
		test_name = "s_{0}_b_{1}".format(i, j)
		param_updates = {
			"test_name": test_name,
			"switch_error": s,
			"blip_error": b,
		}
		params = params_base.copy()
		params.update(param_updates)
		params.update(data_info)
		params_path = os.path.join(params_dir, test_name + ".pickle")
		disp.add_job(out_dir, params_path, params, num_trials)

if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))

	script_path = os.path.join(curr_path, "sim_test.py")
	batch_size = 10
	num_trials = 50

	disp = Dispatcher(script_path, batch_size)

	data_info = {
		"vcf_dir": "/agusevlab/awang/job_data/sim_coloc/vcfs/",
		"vcf_name_template": "ALL.{0}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz",
		"sample_filter_path": "/agusevlab/awang/job_data/sim_coloc/vcfs/integrated_call_samples_v3.20130502.ALL.panel",
		"snp_filter_path": "/agusevlab/awang/job_data/sim_coloc/1000g/snp_filter.pickle"
	}
	params_dir = "/agusevlab/awang/job_data/sim/params/"
	if not os.path.exists(params_dir):
		os.makedirs(params_dir)
	out_dir_base = "/agusevlab/awang/job_data/sim/outs/"

	std_al_dev = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
	coverage = [10, 20, 50, 100, 500, 1000]
	test_dev_cov(
		disp, 
		data_info,
		params_dir, 
		out_dir_base, 
		std_al_dev,
		coverage, 
		num_trials,
		script_path
	)

	disp.submit()