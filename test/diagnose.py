import sys
import os
import subprocess

class Logger(object):
	def __init__(self, logfile):
		self.terminal = sys.stdout
		self.log = open(logfile, "w")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		pass

def diagnose_nodes(nodelist, testfile, logfile):
	sys.stdout = Logger(logfile)
	good_nodes = []
	for n in nodelist:
		print("TEST {0}".format(n))
		job_args = [
			"srun", 
			"--export=ALL,PYTHONVERBOSE=2",
			"-w",
			n,
			"--mem",
			"10000",
			testfile,
		]
		print("TRY " + " ".join(job_args))
		try:
			output = subprocess.check_output(job_args, timeout=30, stderr=subprocess.STDOUT)
			print(output.decode('UTF-8'))
			good_nodes.append(n)
			print("COMPLETE")
		except subprocess.TimeoutExpired as e:
			print(e.output.decode('UTF-8'))
			print("TERMINATED")
		print("----------------------")
		print("")
	print("GOOD NODES:")
	print(", ".join(good_nodes))

if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))
	nodelist = ["node{0:02d}".format(i) for i in range(1, 20)]
	testfile = os.path.join(curr_path, "node_test.py")
	logfile = "/agusevlab/awang/nodetest.txt"
	diagnose_nodes(nodelist, testfile, logfile)

