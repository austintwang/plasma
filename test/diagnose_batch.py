import sys
import os
import subprocess

def diagnose_sbatch(count, testfile, outdir):
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	for j in range(count):
		job_args = [
			"sbatch", 
			"--export=ALL,PYTHONVERBOSE=2",
			"--mem",
			"10000",
			"-o",
			os.path.join(outdir, "{0:03d}.out".format(j)),
			testfile,
		]
		submission = subprocess.run(job_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		print(str(submission.stdout, 'utf-8').rstrip())

if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))
	testfile = os.path.join(curr_path, "node_test.py")
	outdir = "/agusevlab/awang/batchtest"
	diagnose_sbatch(100, testfile, logfile)

