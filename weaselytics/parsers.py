# coding: utf-8
"""
Parser for common spectral data.

`ParsedData` reads a two-column text file by matching a leading pair of
numbers on each line, which skips instrument headers without needing to
know their format.
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
        Read the file and extract its two numeric columns.

        Scans every line for a leading pair of numbers, keeping the
        lines that match and skipping the rest, so instrument headers
        and trailing metadata need no separate handling.

        Parameters
        ----------
        path : str
            Path of the file to be parsed.

        Returns
        -------
        data : array-like, shape (2,N)
            Array containing the x-y data extracted from the file.

        Notes
        -----
        A line qualifies when it begins with two whitespace-separated
        numbers, each optionally signed, with an optional decimal part
        and an optional complete exponent. Requiring the exponent to be
        complete is what keeps ``1.5e`` out while still admitting a
        bare integer such as the ``0`` timestamp.

        Anything after the first two columns is ignored, and a file
        whose columns are separated by something other than whitespace
        yields shape ``(2, 0)`` rather than an error. See the README
        TO DO on generalising the parser.

        """
        xlist = list()
        ylist = list()
        # A number is digits, an optional decimal part, and an optional
        # complete exponent. Making the exponent parts individually
        # optional either rejects bare integers like the t=0 timestamp
        # (if the trailing digits are required) or accepts malformed
        # values like '1.5e' (if they are not).
        number = r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"
        pattern = rf"^({number})\s+({number})"

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

