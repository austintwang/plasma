#!/usr/bin/env python3

print("PYTHON STARTED")

import sys
import numpy

print("IMPORTS FINISHED")

node = sys.argv[1]
test_num = sys.argv[2]

with open("/agusevlab/awang/batchtest/summary.txt", "a") as outfile:
	outfile.write("NODE {0}, TEST {1:02d} COMPLETE\n".format(node, int(test_num)))

print("COMPLETE")