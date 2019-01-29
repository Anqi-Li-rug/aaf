Go to the directory which contains  'AAF.py' and 'Coeffs16384Kaiser-quant.dat'

To run the code in python: 
>import AAF
>AAF.MS_corr('MeasurementSet name', tolerance) : 
#tolerance is a limit, filter response below which will be ignored. 
example:
>AAF.MS_corr('WSRTA18017026_B00.MS/',0.00001)


