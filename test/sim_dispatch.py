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
		# raise Exception ####
		for i in self.jobs:
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
					
def test_dev_cov_display():
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
		"primary_var_display": "Standard Allelic Deviation",
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

def test_dev_cov():
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
		"primary_var_display": "AS Variance (Standard Allelic Deviation)",
		"secondary_var": "coverage",
		"secondary_var_display": "Mean Coverage",
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

def test_dev_herit():
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
		"primary_var_display": "AS Variance (Standard Allelic Deviation)",
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

def test_multi_cv():
	params = {
		"num_snps": 100,
		"num_ppl": 95,
		"overdispersion": 0.05,
		"herit_eqtl": 0.05,
		"herit_ase": 0.4,
		"herit_ase_manual": 0.1,
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
		"test_count": 1,
		"test_name": "multi_cv",
		"test_path": "/home/austin/Documents/Gusev/Results/ase_finemap_results/Simulations",
		"iterations": 500,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase"]) 
	}
	tests = [2]
	bm = Benchmark(params)
	for t in tests:
		it = int(spm.comb(params["num_snps"], t))
		bm.test(max_causal=t+1, num_causal=t, search_iterations=it)
	bm.output_summary()

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
		"region_size": 200000,
		"max_snps": 1000,
		"num_samples_qtl": None,
		"num_samples_gwas": None,
		"maf_thresh": 0.01,
		"overdispersion": 0.05,
		"herit_qtl": 0.05,
		"herit_as": 0.4,
		"herit_gwas": None,
		"std_al_dev": 0.7,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"min_causal": 0,
		"max_causal": 1,
		"test_name": None,
		"confidence": 0.95,
		"model_flavors": set(["indep", "eqtl", "ase", "ecav"])
	}
	out_dir = os.path.join(out_dir_base, "shared")
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

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

if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))

	script_path = os.path.join(curr_path, "coloc_test.py")
	batch_size = 10
	num_trials = 50

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

	# qtl_sizes = [10, 50, 100, 200, 500]
	# gwas_sizes = [10000, 50000, 100000, 200000, 500000]

	qtl_sizes = [1000, 500, 200, 100, 50, 10]
	gwas_sizes = [500000, 200000, 100000, 50000, 10000]

	# gwas_herits = [.01/100, .05/1000]
	gwas_herits = [0.001, 0.0001]
	# gwas_herits = [.1] ####
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

	ld_thresh = [0., 0.2, 0.4, 0.8, 0.95]
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