#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals 
from __future__ import absolute_import

from .ase_finemap import Finemap

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='foo help')
args = parser.parse_args()