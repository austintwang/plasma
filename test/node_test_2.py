#!/usr/bin/env python3

print("PYTHON STARTED")

import sys
import os
import random
import traceback
import pickle
import signal
from contextlib import contextmanager

print("BUILTINS IMPORTS FINISHED")

# import numpy as np
# import pandas as pd

# print("USER IMPORTS FINISHED")

if __name__ == '__main__' and __package__ is None:
	__package__ = 'test'
	import sys
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
	sys.path.insert(0, "/agusevlab/awang/plasma")

from . import Finemap, LocusSimulator, Caviar, CaviarASE, FmBenner, Rasqual

print("PACKAGE IMPORTS FINISHED")

# np.array([1,2,3])

node = sys.argv[1]
test_num = sys.argv[2]

with open("/agusevlab/awang/batchtest/summary.txt", "a") as outfile:
	outfile.write("NODE {0}, TEST {1:02d} COMPLETE\n".format(node, int(test_num)))

print("COMPLETE")