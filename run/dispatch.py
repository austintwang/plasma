from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import time
import drmaa
# try:
# 	import subprocess32 as subprocess
# except ImportError:
# 	import subprocess
try:
	import cpickle as pickle
except ImportError:
	import pickle

# LOCAL = False

def dispatch(s, target, output_path, input_path, params_path, script_path):
	job_input_path = os.path.join(input_path, target)
	job_output_path = os.path.join(output_path, target)

	stdout_path = os.path.join(job_output_path, "stdout.txt")
	stderr_path = os.path.join(job_output_path, "stderr.txt")

	jt = s.createJobTemplate()
	jt.remoteCommand = script_path
	jt.args = [job_output_path, job_input_path, params_path]
	jt.joinFiles = True
	jt.outputPath = stdout_path
	st.errorPath = stderr_path

	job_id = s.runJob(jt)

	s.deleteJobTemplate(jt)

	return job_id


	# print("dispatch") ####

	# try:
	# 	args = ["qsub", "-e", stderr_path, "-o", stdout_path, "-v", "DATA_DIR=\""+target_dir+"\"", script_path]
	# 	# args = [script_path, target_dir] ####
	# 	# print(args) ####
	# 	job_info = subprocess.check_output(args)
	# 	# print(job_info) ####
	# 	if LOCAL:
	# 		job_id = job_info.rstrip()
	# 	else:
	# 		job_id = job_info.split()[2]

	# 	return job_id
	# except subprocess.CalledProcessError as e:
	# 	raise e
	# 	return False

def poll(s, job_id):
	status = s.jobStatus(job_id)
	if status in [drmaa.JobState.DONE, drmaa.JobState.UNDETERMINED]:
		return "complete"
	elif status == drmaa.JobState.FAILED:
		return "failed"
	return "in_progress"

	# args = ["qstat", "-f", job_id]
	# job_info = subprocess.check_output(args)
	# # print(job_info) ####
	# exit_code = None
	# lines = job_info.split("\n")
	# for l in lines:
	# 	stat = l.strip().split(" = ")
	# 	if stat[0] == "job_state":
	# 		state = stat[1]
	# 	if stat[0] == "exit_status":
	# 		exit_code = stat[1]

	# print(job_id, state, exit_code) ####
	# return state, exit_code


	# if len(lines) > 2:
	# 	header = [i.lower() for i in lines[0]]
	# 	if LOCAL:
	# 		state_idx = header.index("s")
	# 	else:
	# 		state_idx = header.index("state")
	# 	job_state = lines[2][state_idx]
	# 	# print(lines[2]) ####
	# 	raise Exception ####
	# 	return job_state
	# raise Exception
	# return None

def delete(s, job_id):
	s.control(job_id, drmaa.JobControlAction.TERMINATE)
	
	# args = ["qdel", job_id]

def run(output_path, input_path, params_path, hyperparams, num_tasks, poll_freq, script_path):
	with open(params_path, "wb") as params_file:
		pickle.dump(hyperparams, params_file)

	with drmaa.Session() as s:
		targets = os.listdir(input_path)

		wait_pool = set(targets)
		active_pool = {}
		complete_pool = set()
		fail_pool = set()
		dead_pool = set()
		# print(wait_pool) ####
		try:
			while (len(wait_pool) > 0) or (len(active_pool) > 0):
				# print("woiehoifwe") ####
				to_remove = set()
				for k, v in active_pool.viewitems():
					state = poll(s, v)
					# print(state) ####
					if state == "failed":
						delete(s, v)
						fail_pool.add(k)
						to_remove.add(k)
						# active_pool.pop(k)
					# elif exit_code and exit_code != 0:
					# 	# print(exit_code) ####
					# 	fail_pool.add(k)
					# 	to_remove.add(k)
					elif state == "complete":
						complete_pool.add(k)
						to_remove.add(k)
						# active_pool.pop(k)
				for i in to_remove:
					active_pool.pop(i)

				vacant = num_tasks - len(active_pool)
				for _ in xrange(vacant):
					if len(wait_pool) == 0:
						break
					target = wait_pool.pop()
					job_id = dispatch(s, target, output_path, input_path, params_path, script_path)
					active_pool[target] = job_id

				time.sleep(poll_freq)

			while (len(fail_pool) > 0) or (len(active_pool) > 0):
				to_remove = set()
				for k, v in active_pool.viewitems():
					state = poll(s, v)
					if state == "failed":
						delete(s, v)
						dead_pool.add(k)
						to_remove.add(k)
					# elif exit_code and exit_code != 0:
					# 	# print(exit_code) ####
					# 	dead_pool.add(k)
					# 	to_remove.add(k)
					elif state == "complete":
						complete_pool.add(k)
						to_remove.add(k)

				for i in to_remove:
					active_pool.pop(i)

				vacant = num_tasks - len(active_pool)
				for _ in xrange(vacant):
					if len(fail_pool) == 0:
						break
					target = fail_pool.pop()
					job_id = dispatch(s, target, output_path, input_path, params_path, script_path)
					active_pool[target] = job_id

				time.sleep(poll_freq)

		finally:
			for v in active_pool.values():
				delete(v)



if __name__ == '__main__':
	curr_path = os.path.abspath(os.path.dirname(__file__))

	# # Test Run
	# out_dir = os.path.join(curr_path, "test_results")
	# script_path = os.path.join(curr_path, "job.py")
	# hyperparams = {
	# 	"overdispersion": 0.05,
	# 	"prop_noise_eqtl": 0.95,
	# 	"prop_noise_ase": 0.50,
	# 	"std_fraction": 0.75,
	# 	"min_causal": 1,
	# 	"num_causal": 1,
	# 	"coverage": 100,
	# 	"search_mode": "exhaustive",
	# 	"max_causal": 1,
	# 	"confidence": 0.95, 
	# 	"max_ppl": 100
	# }

	# run(
	# 	out_dir, 
	# 	30000, 
	# 	hyperparams, 
	# 	7, 
	# 	1, 
	# 	script_path, 
	# 	"test_run",
	# 	parse_input=False
	# )

	# Kidney Data
	output_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ_ASVCF/outs/1cv_all"
	input_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ_ASVCF/jobs"
	params_path = "/bcb/agusevlab/awang/job_data/KIRC_RNASEQ_ASVCF/params/1cv_all.pickle"
	script_path = os.path.join(curr_path, "job.py")
	hyperparams = {
		"overdispersion": 0.05,
		"prop_noise_eqtl": 0.95,
		"prop_noise_ase": 0.50,
		"std_fraction": 0.75,
		"min_causal": 1,
		"num_causal": 1,
		"coverage": 100,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"confidence": 0.95, 
		"max_ppl": 100
	}

	num_tasks = 500
	poll_freq = 10

	run(
		output_path, 
		input_path, 
		params_path, 
		hyperparams, 
		num_tasks, 
		poll_freq, 
		script_path
	)





