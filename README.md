# `Weaselytics`

Python package to extract and analyse chromatographic data.

<!-- ![show](examples/show-use3.gif) -->

A Python 3 script for (hassle-free) plotting of HPLC chromatograms from output files with baseline correction, peak detection and retention time determination through curve fitting.

### Quick start

 Start the script with:

```console
python3 weaselytics.py [OPTION] filename
```

it will show the chromatogram... (to complete)

### Command-line options

- `filename` , required: filename (.txt)
- `-s` , optional: shows the `matplotlib` window(s)
- `-p` , optional: prints the `matplotlib` window(s)
- `-e` , optional: exports the baseline corrected data to `filename_bl.txt`
- `-o` , optional: outputs the data to `filename.csv`
- `-os` `str` , optional: outputs the fitted data and statistiques for the peak labeled `<ARG>` to `filename_<ARG>.csv`
- `-n` , optional: does not try to fit a chromatographic peak
- `-nb` , optional: does not try to baseline correct the chromatogram
- `-ns` , optional: does not try to smooth the chromatogram
- `-x0`  `float` , optional: starts fitting procedure at `x0` min (`x0 > 0`)
- `-x1`  `float` , optional: ends fitting procedure at `x1` min (`x1 > x0 > 0`)

### Script options

<!--
There are numerous ways to configure the spectrum in the script:
Check `# plot config section - configure here` in the script. 
You can even configure the script to plot of the single line shape functions.
-->
(**TO UPDATE**)

### Code options


<!--
Colors, line thickness, line styles, level of peak detection and 
more can be changed in the code directly.
-->
(**TO UPDATE**)

### Remarks

<!--
The SVG file will be replaced everytime you start the script with the same output file. 
If you want to keep the file, you have to rename it. 
The data are taken from the section "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS".
-->
(**TO UPDATE**)

## Examples:

<!--
![Example 1](examples/example1.png)
![Example 2](examples/example2.png)
![Example 3](examples/example3.png)
-->

## Requirements
- `re`

- `sys`

- `os`

- `argparse`

- `numpy`

- `pandas`
  
- `matplotlib`

- `seaborn`

- `scipy`

- `pybaselines` (development branch)

- `numba` (recommanded)

<!-- - `statsmodels` -->

- `time`

## Contributor

Contributed by Emmanuel Bourret

## TO TO

- Refactor the main code into a package.

- Add smoothing function before the baseline correction.

- Write a more general parser.

