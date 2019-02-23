from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import drmaa

# import gzip
# import sys
# try:
# 	import cPickle as pickle
# except ImportError:
# 	import pickle

def dispatch(s, script_path, chr_path, bed_path, out_dir, margin, chr_num):
	stdout_path = ":" + os.path.join(out_dir, "job_reports", "{0}_stdout.txt".format(chr_num))
	stderr_path = ":" + os.path.join(out_dir, "job_reports", "{0}_stderr.txt".format(chr_num))

	environ = os.environ
	environ["SGE_O_SHELL"] = "/bcb/agusevlab/awang/python/bin/python"

	jt = s.createJobTemplate()
	jt.remoteCommand = script_path
	jt.args = [chr_path, bed_path, out_dir, margin, chr_num]
	jt.outputPath = stdout_path
	jt.errorPath = stderr_path
	jt.jobEnvironment = environ

	job_id = s.runJob(jt)

	s.deleteJobTemplate(jt)

	return job_id

def make_targets(script_path, chr_info, bed_path, out_dir, margin):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	reports_path = os.path.join(out_dir, "job_reports")
	if not os.path.exists(reports_path):
		os.makedirs(reports_path)

	with drmaa.Session() as s:
		joblist = []
		for k, v in chr_info.viewitems():
			job_id = dispatch(s, script_path, v, bed_path, out_dir, margin, k)
			joblist.append(job_id)

		s.synchronize(joblist, drmaa.Session.TIMEOUT_WAIT_FOREVER, True)


if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))
	script_path = os.path.join(curr_path, "make_chr.py")

	# # Test Run
	# chr_dir_test = os.path.join(curr_path, "test_data", "chrs")
	# chr_paths = [chr_dir_test + "KIRC.ALL.AS.chr22.vcf"]
	# bed_path_test = os.path.join(curr_path, "test_data", "test_22.bed")
	# out_dir = os.path.join(curr_path, "test_results")
	# script_path = os.path.join(curr_path, "job.py")
	# # hyperparams = {
	# # 	"overdispersion": 0.05,
	# # 	"prop_noise_eqtl": 0.95,
	# # 	"prop_noise_ase": 0.50,
	# # 	"std_fraction": 0.75,
	# # 	"min_causal": 1,
	# # 	"num_causal": 1,
	# # 	"search_mode": "exhaustive",
	# # 	"max_causal": 1,
	# # 	"confidence": 0.95, 
	# # 	"max_ppl": 100
	# # }

	# make_targets(
	# 	chr_paths, 
	# 	bed_path_test, 
	# 	out_dir, 
	# 	30000, 
	# )

	# # Kidney Data
	# chr_dir = "/bcb/agusevlab/DATA/KIRC_RNASEQ/ASVCF"
	# # chrs = ["KIRC.ALL.AS.chr{0}.vcf.gz".format(i + 1) for i in xrange(22)]
	# # chr_paths = [os.path.join(chr_dir, i) for i in chrs]
	# chr_info = {
	# 	i + 1: os.path.join(chr_dir, "KIRC.ALL.AS.chr{0}.vcf.gz".format(i + 1)) for i in xrange(22)
	# }
	# bed_path = "/bcb/agusevlab/DATA/KIRC_RNASEQ/ASVCF/gencode.protein_coding.transcripts.bed"
	# out_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ"

	# make_targets(
	# 	script_path,
	# 	chr_info, 
	# 	bed_path, 
	# 	out_dir, 
	# 	100000, 
	# )

	# Prostate Data
	chr_dir = "/bcb/agusevlab/DATA/KIRC_RNASEQ/ASVCF"
	# chrs = ["KIRC.ALL.AS.chr{0}.vcf.gz".format(i + 1) for i in xrange(22)]
	# chr_paths = [os.path.join(chr_dir, i) for i in chrs]
	chr_info = {
		i + 1: os.path.join(chr_dir, "KIRC.ALL.AS.chr{0}.vcf.gz".format(i + 1)) for i in xrange(22)
	}
	bed_path = "/bcb/agusevlab/DATA/KIRC_RNASEQ/ASVCF/gencode.protein_coding.transcripts.bed"
	out_dir = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ"

	make_targets(
		script_path,
		chr_info, 
		bed_path, 
		out_dir, 
		100000, 
	)