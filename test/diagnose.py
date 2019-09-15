import signal
from contextlib import contextmanager
import sys
import os
import subprocess

class TimeoutException(Exception): 
	pass

class Logger(object):
	def __init__(self, logfile):
		self.terminal = sys.stdout
		self.log = open(logfile, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		pass

# @contextmanager
# def time_limit(seconds):
# 	def signal_handler(signum, frame):
# 		raise TimeoutException()
# 	signal.signal(signal.SIGALRM, signal_handler)
# 	signal.alarm(seconds)
# 	try:
# 		yield
# 	finally:
# 		signal.alarm(0)

def diagnose_nodes(nodelist, testfile, logfile):
	sys.stdout = Logger(logfile)
	for n in nodelist:
		print("TEST {0}".format(n))
		job_args = [
			"srun", 
			"-w",
			n,
			"--mem",
			"10000",
			"--pty",
			testfile,
		]
		print("TRY " + " ".join(job_args))
		try:
			output = subprocess.check_output(job_args, timeout=15)
			print(output.decode('UTF-8'))
		except subprocess.TimeoutExpired as e:
			print(e.output.decode('UTF-8'))
			print("TERMINATED")
		print("----------------------")
		print("")

if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))
	nodelist = ["node{0}".format(i) for i in range(20)]
	testfile = os.path.join(curr_path, "node-test.py")
	logfile = "/agusevlab/awang/nodetest.txt"
	diagnose_nodes(nodelist, testfile, logfile)

