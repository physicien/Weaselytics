#!/usr/bin/python
# coding: utf-8
"""
Parser for common spectral data.
"""

#import os
import re
from pathlib import Path
import numpy as np

class ParsedData:
    """
    """

    def __init__ (self, path):
        self.path = path
        self.raw = self.read_file()
        self.data = self.read_data()

    def read_file(self):
        with open(self.path,'r') as f:
            lines = [line.rstrip() for line in f]
        return lines

    def read_data(self):
        xlist = list()
        ylist = list()
        pattern = r"^([+-]?\d+\.?\d*[eE]?[-+]?\d*)\s+" + \
                    r"([+-]?\d+\.?\d*[eE]?[-+]?\d*)"
        for line in self.raw:
            if re.search(pattern, line):
                xlist.append(float(line.strip().split('\t')[0]))
                ylist.append(float(line.strip().split('\t')[1]))
        out = np.array([xlist, ylist])
        return out

