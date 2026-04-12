#!/usr/bin/python
# coding: utf-8
"""
Parser for common spectral data.
"""

import re
from pathlib import Path
import numpy as np

class ParsedData:
    """
    Parse a file containing x-y data organized in two columns.

    Parameters
    ----------
    path : str
        Path of the file to be parsed.

    Attributes
    ----------
    path : str
        Path of the file to be parsed.
    data : array-like, shape (2,N)
        Array containing the x-y data extracted from the file.

    NOTE
    ----
    Do I really need to keep `self.path`?

    """

    def __init__ (self, path):
        self.path = path
        self.data = self.read_data(path)

    def read_data(self, path):
        """
        Read the file and

        Parameters
        ----------
        path : str
            Path of the file to be parsed.

        Returns
        -------
        data : array-like, shape (2,N)
            Array containing the x-y data extracted from the file.

        """
        xlist = list()
        ylist = list()
        pattern = r"^([+-]?\d+\.?\d*[eE]?[-+]?\d*)\s+" + \
                    r"([+-]?\d+\.?\d*[eE]?[-+]?\d*)"

        # Read the file
        with open(path, 'r') as f:
            lines = [line.rstrip() for line in f]

        # Extract data from each line
        for line in lines:
            if re.match(pattern, line):
                xy = re.split("\s+", line)
                xlist.append(float(xy[0]))
                ylist.append(float(xy[1]))
        data = np.array([xlist, ylist])
        return data

