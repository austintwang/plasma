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
		"num_snps": 200,
		"num_ppl": 95,
		"var_effect_size": 2.0,
		"overdispersion": 0.05,
		"prop_noise_eqtl": 0.50,
		"prop_noise_ase": 0.0,
		"baseline_exp": 4.0,
		"num_causal": 1,
		"ase_read_prop": 0.25,
		"search_mode": "exhaustive",
		"max_causal": 1,
		"search_iterations": 50,
		"primary_var": "var_effect_size",
		"primary_var_display": "Variance of Simulated Effect Sizes",
		"test_count": 4,
		"test_name": "dummy_test",
		"iterations": 1,
		"confidence": 0.95
	}
	tests = [25.0,]
	bm = Benchmark(params)
	for t in tests:
		bm.test(var_effect_size=t)
	test.output_summary()

if __name__ == "__main__":
	dummy_test()
