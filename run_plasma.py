#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

if __name__ == '__main__' and __package__ is None:
	__package__ = 'plasma'
	from ase_finemap import Finemap
else:
	from .ase_finemap import Finemap

import numpy as np
import os
import argparse

def run_plasma(args):
	params = {}

	params["hap_A"] = np.loadtxt(args.hap_A_path)
	params["counts_A"] = np.loadtxt(args.counts_A_path)
	params["hap_B"] = np.loadtxt(args.hap_B_path)
	params["counts_B"] = np.loadtxt(args.counts_B_path)
	params["total_exp"] = np.loadtxt(args.total_exp_path)

	if overdispersion_path is not None:
		params["overdispersion"] = np.loadtxt(args.hap_A_path)
	else:
		params["overdispersion"] = args.overdispersion_global

	if args.as_only and args.qtl_only:
		raise ValueError("PLASMA cannot be in both AS-Only and QTL-Only modes")
	else:
		params["as_only"] = args.as_only
		params["qtl_only"] = args.qtl_only

	model = Finemap(**params)
	model.initialize()

	if args.search_mode == "exhaustive":
		model.search_exhaustive(1, args.max_causal)
	elif args.search_mode == "shotgun":
		model.search_shotgun(1, args.max_causal, 0.001, 1000, 100000)
	else:
		raise ValueError("Invalid search mode specified")

	credible_set = model.get_causal_set(args.confidence)
	ppas = model.get_ppas()
	
	cset_path = os.path.join(args.out_dir, "cset.txt")
	ppas_path = os.path.join(args.out_dir, "ppas.txt")

	np.savetxt(cset_path, credible_set)
	np.savetxt(ppas_path, ppas)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="PLASMA Quickrun", epilog="See README.md for more information")

	parser.add_argument('hap_A_path', help="Path to haplotype A genotypes file")
	parser.add_argument('counts_A_path', help="Path to haplotype A mapped counts file")
	parser.add_argument('hap_B_path', help="Path to haplotype B genotypes file")
	parser.add_argument('counts_B_path', help="Path to haplotype B mapped counts file")
	parser.add_argument('out_dir', help="Path to output directory")
	
	parser.add_argument('--total_exp_path', '-t', help="Path to total QTL phenotype file (Default: Sum of counts files)")
	parser.add_argument('--overdispersion_path', '-o', help="Path to individual-level AS overdispersion file (Default: Global overdispersion)")
	parser.add_argument('--overdispersion_global', '-g', default=0., help="Global AS overdispersion (Default: 0)")
	parser.add_argument('--as_only', '-a', action='store_true', help="AS-Only Mode")
	parser.add_argument('--qtl_only', '-q', action='store_true', help="QTL-Only Mode")
	parser.add_argument('--search_mode', '-s', default="exhaustive", help="Causal configuration search mode (Default: \"exhaustive\")")
	parser.add_argument('--max_causal', '-m', default=1, help="Maximum number of causal configurations searched (Default: 1)")
	parser.add_argument('--confidence', '-c', default=0.95, help="Credible set confidence level (Default: 0.95)")
	
	args = parser.parse_args()

	run_plasma(args)