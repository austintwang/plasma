from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import random
import pickle
try:
	import cpickle as pickle
except ImportError:
	import pickle

class Haplotypes(object):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	data_path = os.path.join(dir_path, "haplotypes")
	hap_name = "CEU.sampled_haplotypes"
	hap_path = os.path.join(data_path, hap_name)
	pickle_name = hap_name + ".pickle"
	pickle_path = os.path.join(data_path, pickle_name)
	NUM_SNPS = 1000
	NUM_HAPS = 100
	def __init__(self):
		try:
			with open(self.pickle_path, "rb") as hapfile:
				self.haps = pickle.load(hapfile)
		except StandardError:
			self.reload()
			
	def reload(self):
		self.haps = {}
		self.hap_files = os.listdir(self.hap_path)
		# print(self.hap_files) ####
		for f in self.hap_files:
			if f.endswith(".haps"):
				with open(os.path.join(self.hap_path, f)) as hap:
					hapstr = hap.read()
				# print(hapstr) ####
				# print([[int(j) for j in i.strip().split("\t")] for i in hapstr.strip().split("\n")]) ####
				hap_arr = np.array(
					[[int(j) for j in i.strip().split("\t")] for i in hapstr.strip().split("\n")]
				).T
				# print(hap_arr) ####
				np.place(hap_arr, hap_arr==2, 0)
				if hap_arr.shape[1] == self.NUM_SNPS:
					self.haps[f] = hap_arr
		with open(self.pickle_path, "wb") as hapfile:
			pickle.dump(self.haps, hapfile)

	def draw_haps(self):
		num_ppl = int(self.NUM_HAPS / 2)
		locus = random.choice(list(self.haps))
		locus_haps = self.haps[locus]
		a_ind = np.random.choice(self.NUM_HAPS, num_ppl, replace=False)
		a_set = set(a_ind)
		b_ind = np.array([i for i in xrange(self.NUM_HAPS) if i not in a_set])
		hapA = locus_haps[a_ind]
		hapB = locus_haps[b_ind]
		np.random.shuffle(hapA)
		np.random.shuffle(hapB)
		# print(a_ind) ####
		# print(b_ind) ####
		# # print(locus_haps) ####
		# haps_shuffled = np.random.shuffle(locus_haps)
		# # print(haps_shuffled) ####
		# hapA = haps_shuffled[:self.NUM_PPL]
		# hapB = haps_shuffled[self.NUM_PPL:]
		return hapA, hapB

