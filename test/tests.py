from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

import numpy as np
import os
from datetime import datetime
import seaborn as sns
import pandas as pd

from . import Benchmark
from . import Haplotypes

def dummy_test():
	params = {
		"num_snps": 1000,
		"num_ppl": 50,
		"var_effect_size": 2.0,
		"overdispersion": 0.05,
		"prop_noise": 0.95,
		"baseline_exp": 4.0,
		"num_causal": 1,
		"ase_read_prop": 1.0,
		"search_mode": "shotgun",
		"search_iterations": 10000,
		"primary_var": "var_effect_size",
		"primary_var_display": "Variance of Simulated Effect Sizes",
		"test_count": 4,
		"test_name": "dummy_test",
		"iterations": 10
	}
	tests = [2.0, 5.0, 10.0, 20.0]
	bm = Benchmark(params)
	for t in tests:
		bm.test(var_effect_size=t)
	test.output_summary()

if __name__ == "__main__":
	dummy_test()
