# Apertif Anti-Aliasing Filter

This code will apply an anti-aliasing filter to Apertif data. The code is written by Anqi Li, based on work by Stefan Wijnholds and Sebastiaan van der Tol.

To use the code, use the stand-alone mode:

```bash
./AAF.py --help
usage: AAF.py [-h] [-t TOLERANCE] [-o OUTPUT_COLUMN] msname

Apertif Anti-aliasing filter.

positional arguments:
  msname                Name of Measurement Set

optional arguments:
  -h, --help            show this help message and exit
  -t TOLERANCE, --tolerance TOLERANCE
                        Filter response below this limit will be ignored
  -o OUTPUT_COLUMN, --output-column OUTPUT_COLUMN
                        Column to output the corrected visibilities to
                        (default DATA_AAF)
```

Also, it can be used from python: go to the directory which contains  'AAF.py' and 'Coeffs16384Kaiser-quant.dat'

To run the code in python: 
```python
import AAF
AAF.MS_corr('WSRTA18017026_B00.MS/',0.00001)
```