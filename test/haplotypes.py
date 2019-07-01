import numpy as np
import os
import random
try:
	import pickle as pickle
except ImportError:
	import pickle

class Haplotypes(object):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	data_path = os.path.join(dir_path, "haplotypes")
	hap_name = "CEU.sampled_haplotypes"
	hap_path = os.path.join(data_path, hap_name)
	pickle_name = hap_name + ".pickle"
	pickle_path = os.path.join(data_path, pickle_name)
	num_haps = 190

	haps = None
	num_snps_total = None

	def __init__(self, params):
		if Haplotypes.haps is None:
			Haplotypes.load()

		self.num_snps = params["num_snps"]
		self.num_ppl = params["num_ppl"]
		if self.num_ppl > Haplotypes.num_haps // 2:
			raise ValueError("Not enough haplotypes to generate genotypes")
			
	@classmethod
	def load(cls):
		try:
			with open(cls.pickle_path, "rb") as hapfile:
				cls.haps = pickle.load(hapfile)
		except Exception:
			# print("iosheiof") ####
			cls.build()
		finally:
			# print(cls.haps) ####
			cls.num_snps_total = cls.haps.shape[0]

	@classmethod
	def build(cls):
		# print("wheiieieiieiei") ####
		cls.hap_files = os.listdir(cls.hap_path)
		haps_list = [] 
		for f in cls.hap_files:
			# print("f") ####
			if f.endswith(".haps"):
				with open(os.path.join(cls.hap_path, f)) as hap:
					hapstr = hap.read()
				hap_block = [
					[int(j) for j in i.strip().split("\t")] 
					for i in hapstr.strip().split("\n")
				]
				for s in hap_block:
					if all(0 <= i <= 1 for i in s):
						prop_1 = sum(s) / len(s)
						if 0.01 <= prop_1 <=0.99:
							# print(len(s)) ####
							# print(prop_1) ####
							haps_list.append(s)
		cls.haps = np.array(haps_list)
		# print(cls.haps) ####
		with open(cls.pickle_path, "wb") as hapfile:
			pickle.dump(cls.haps, hapfile)
		# print("wheifhwoeihfowe") ####
		

		# self.haps = {}
		# self.hap_files = os.listdir(self.hap_path)
		# # print(self.hap_files) ####
		# for f in self.hap_files:
		# 	if f.endswith(".haps"):
		# 		with open(os.path.join(self.hap_path, f)) as hap:
		# 			hapstr = hap.read()
		# 		# print(hapstr) ####
		# 		# print([[int(j) for j in i.strip().split("\t")] for i in hapstr.strip().split("\n")]) ####
		# 		hap_arr = np.array(
		# 			[[int(j) for j in i.strip().split("\t")] for i in hapstr.strip().split("\n")]
		# 		).T
		# 		# print(hap_arr) ####
		# 		np.place(hap_arr, hap_arr>1, 0)
		# 		if hap_arr.shape[1] == self.NUM_SNPS_RAW:
		# 			self.haps[f] = hap_arr
		# with open(self.pickle_path, "wb") as hapfile:
		# 	pickle.dump(self.haps, hapfile)

	def draw_haps(self):
		start = np.random.randint(0, high=Haplotypes.num_snps_total-self.num_snps)
		end = start + self.num_snps
		section = np.arange(start, end)
		haps_section = Haplotypes.haps[section]

		locus_haps = haps_section.T

		# num_ppl = int(self.NUM_HAPS / 2)
		# locus = random.choice(list(self.haps))
		# locus_haps = self.haps[locus]
		a_ind = np.random.choice(Haplotypes.num_haps, self.num_ppl, replace=False)
		a_set = set(a_ind)
		b_ind = np.array([i for i in range(Haplotypes.num_haps) if i not in a_set])
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

