import sys
import os
import subprocess
import shutil

def diagnose_sbatch(count_per_node, nodelist, testfile, outdir):
	job_ids = []
	if os.path.exists(outdir):
		shutil.rmtree(outdir, ignore_errors=True)
	os.makedirs(outdir)
	for n in nodelist:
		for j in range(count_per_node):
			job_args = [
				"sbatch", 
				"-J",
				"{0}_{1:02d}".format(n, j),
				"-w",
				n,
				"--export=ALL,PYTHONVERBOSE=1",
				"--mem",
				"10000",
				"-o",
				os.path.join(outdir, "{0}_{1:02d}.out".format(n, j)),
				testfile,
				n,
				str(j)
			]
			submission = subprocess.run(job_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			submission_str = str(submission.stdout, 'utf-8').rstrip()
			print(submission_str)
			job_ids.append(submission_str.split()[-1])
	print(",".join(job_ids))

if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))
	testfile = os.path.join(curr_path, "node_test_2.py")
	outdir = "/agusevlab/awang/batchtest"
	nodelist = ["node{0:02d}".format(i) for i in range(1, 20)]
	diagnose_sbatch(10, nodelist, testfile, outdir)

