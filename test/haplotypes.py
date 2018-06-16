from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
import random

class Haplotypes(object):
	hap_path = os.path.join("haplotypes", "CEU.sampled_haplotypes")
	NUM_SNPS= 1000
	def __init__(self):
		self.hap_files = os.listdir(hap_path)
		self.haps = {}
		for f in self.hap_files:
			if f.endswith(".hap"):
				with open(os.path.join(hap_path, f)) as hap:
					hapstr = hap.read()
				hap_arr = np.ndarray(
					[[int(j) for j in i.split("\t")] for i in hapstr.split("\n")]
				).T
				if hap_arr.shape[1] == NUM_SNPS:
					self.haps[f] = hap_arr

	def draw_haps(self):
		locus = random.choice(list(self.haps))
		locus_haps = self.haps[locus]
		haps_shuffled = np.random.shuffle(locus_haps)
		hapA = haps_shuffled[:NUM_PPL]
		hapB = haps_shuffled[NUM_PPL:]
		return hap_A, hapB

