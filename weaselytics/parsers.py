# coding: utf-8
"""
Parser for common spectral data.
"""

import re

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
    data : array-like, shape (2,N)
        Array containing the x-y data extracted from the file.

    """

    def __init__(self, path: str) -> None:
        self.data: np.ndarray = self.read_data(path)

    def read_data(self, path: str) -> np.ndarray:
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
        pattern = r"^([+-]?\d+\.?\d*[eE]?[-+]?\d+)\s+" + \
                    r"([+-]?\d+\.?\d*[eE]?[-+]?\d+)"

        # Read the file
        with open(path, 'r') as f:
            lines = [line.rstrip() for line in f]

        # Extract data from each line
        for line in lines:
            m = re.match(pattern, line)
            if m:
                xlist.append(float(m.group(1)))
                ylist.append(float(m.group(2)))
        data = np.array([xlist, ylist])
        return data

