#!/usr/bin/env python

from __future__ import print_function

from casacore import tables
import numpy as np
import math
import multiprocessing
import itertools
import datetime
import argparse
import os.path


def antialias_visibilities(vis, subband_response, nconv):
    """
    Perform AAF correction

    Args:
        vis (np.array):   input spectrum, which is supposed from raw MS (visibility),
                          datatype complex numbers, shape(num_subbands*64, number of correlations (4: xx, xy, yx, yy)0
        subband_response: filter frequency response, see annotation in the main function
        nconv:            deconvolution length

    Returns:
        np.array: AAF corrected spectrum, same shape and datatype as vis
    """
    # 1. Preparation before implementation AAF

    # Calculate spectrum's subbands numbers, correlator numbers
    ncorr = vis.shape[1]
    num_channels = 64
    num_subbands = vis.shape[0] / 64

    # Calculate the power spectrum, reshape spectrum
    ob_spec = abs(vis)
    ob_spec = np.reshape(ob_spec, (num_subbands, num_channels, ncorr))

    # Estimate missing subbands
    # Find missing subbands and create an array to record the location of missing subbands
    sumspec = np.sum(ob_spec, axis=1)
    zeros = np.where(sumspec == 0)
    flag = np.zeros(ob_spec.shape)
    flag[zeros[0], :, zeros[1]] = 1

    # Use linear interpolation to estimate the missing subbands
    ob_spec1 = list((np.swapaxes(ob_spec, 0, 2)).reshape(ob_spec.shape[1] * ob_spec.shape[2], ob_spec.shape[0]))
    ob_spec2 = np.swapaxes(np.array(
            [np.interp(np.arange(0, ob_spec.shape[0], 1), np.where(x > 0)[0], x[np.where(x > 0)]) for x in
             ob_spec1]).reshape(ob_spec.shape[2], ob_spec.shape[1], ob_spec.shape[0]), 0, 2)
    ob_spec = ob_spec2.copy()

    # 2. Actual AAF calibration:

    # This algorithm also does bandpass correction!!!!!!
    # In case the input spectrum has already been bandpass corrected,  uncomment the next row to undoes this effect.
    # ob_spec = np.swapaxes((np.swapaxes(ob_spec,1,2)/subband_response[1,:]),1,2)

    # Create and initialize a new array to store AAF corrected spectrum
    corr_spec = np.empty_like(ob_spec)
    corr_spec[:] = np.nan

    # Begin with central channel, suppose the central channel of each subband is not influenced by aliasing effect
    corr_spec[:, num_channels / 2, :] = ob_spec[:, num_channels / 2, :] / subband_response[1, num_channels / 2]

    # From central channel downwards,ignore response of previous subband (for a certain subband, channel1 to channel 31
    # was mostly influenced by next subband's channel1 to channel31)
    for chidx in np.arange(num_channels / 2 - 1, 0, -1):
        ratio = -1. * subband_response[2, chidx] / subband_response[1, chidx]
        f_corr = ratio ** (np.arange(nconv - 1, -1, -1)) / subband_response[1, chidx]
        for corri in np.arange(0, ncorr, 1):
            corr_spec[:, chidx, corri] = (np.convolve(ob_spec[:, chidx, corri], f_corr))[nconv - 1:]
            # compensate for missing sample
        if chidx < nconv:
            f_corr = f_corr * ratio
            # estimate of missing data, use neighbouring channel as initial estimate
            ini_spec = corr_spec[:, chidx + 1, :]
            dmissing = np.ones(ncorr)
            for corri in np.arange(0, ncorr, 1):
                # this for-loop is somehow unavoidable, because np.linalg.lstsq only accept one-dimensional array while
                # our data array is too deep.
                dmissing[corri] = np.linalg.lstsq(
                        np.transpose(np.mat(f_corr)),
                        np.transpose(np.mat(ini_spec[num_subbands - nconv:, corri] -
                                            corr_spec[num_subbands - nconv:, chidx, corri])),
                        rcond=None)[0][0, 0]
                corr_spec[num_subbands - nconv:, chidx, corri] = corr_spec[num_subbands - nconv:, chidx, corri] + \
                                                                 np.dot(f_corr, dmissing[corri])

    # From central channel upwards, ignore response of next subband (for a certain subband, channel33 to channel 63
    # were mostly influenced by previous subband's channel33 to channel63)
    for chidx in np.arange(num_channels / 2 + 1, num_channels, 1):
        ratio = -1. * subband_response[0, chidx] / subband_response[1, chidx]
        f_corr = ratio ** (np.arange(0, nconv, 1)) / subband_response[1, chidx]
        for corri in np.arange(0, ncorr, 1):
            corr_spec[:, chidx, corri] = (np.convolve(ob_spec[:, chidx, corri], f_corr))[0:num_subbands]
        if chidx > num_channels - nconv - 1:
            f_corr = f_corr * ratio
            # Estimate of missing data,use neighbouring channel as initial estimate
            ini_spec = corr_spec[:, chidx - 1, :]
            for corri in np.arange(0, ncorr, 1):
                dmissing[corri] = np.linalg.lstsq(
                        np.transpose(np.mat(f_corr)),
                        np.transpose(np.mat(ini_spec[0:nconv, corri] - corr_spec[0:nconv, chidx, corri])),
                        rcond=None)[0][0, 0]
                corr_spec[0:nconv, chidx, corri] = corr_spec[0:nconv, chidx, corri] + np.dot(
                        np.reshape(f_corr, (1, f_corr.size)), dmissing[corri])

    # Dealing with the first channels, since eventually we'll just ignore or flag first channels, so it's okay to
    # disable this block, might save some time
    nedge = 3
    chidx = 0
    ratio = -1. * subband_response[2, chidx] / subband_response[1, chidx]
    f_corr = ratio ** (np.arange(num_subbands - 1, -1, -1)) / subband_response[1, chidx]
    # estimate missing data,use average of first and last channel as initial estimate
    ini_spec = (corr_spec[:, 1, :] + np.roll(corr_spec[:, num_channels - 1, :], 1, axis=0)) / 2.
    ini_spec[0, :] = corr_spec[0, 1, :]
    dmissing = np.ones(ncorr)
    for corri in np.arange(0, ncorr, 1):
        corr_spec[:, chidx, corri] = (np.convolve(ob_spec[:, chidx, corri], f_corr))[np.size(f_corr) - 1:]
    f_corr = ratio * f_corr
    for corri in np.arange(0, ncorr, 1):
        dmissing[corri] = np.linalg.lstsq(np.transpose(np.mat(f_corr[nedge:num_subbands - nedge])),
                                          np.transpose(np.mat(ini_spec[nedge:num_subbands - nedge, corri] -
                                                              corr_spec[nedge:num_subbands - nedge, chidx, corri])),
                                          rcond=None)[0][0, 0]
        corr_spec[:, chidx, corri] = corr_spec[:, chidx, corri] + np.dot(f_corr, dmissing[corri])

    # 3. Flag and reshape AAF corrected spectrum, transform power spectrum back to visibility complex numbers

    # Flag the missing subbands and negative values
    corr_spec[np.where(corr_spec < 0)] = np.nan
    corr_spec[np.where(flag == 1)] = np.nan
    # Reshape spectrum
    corr_spec = np.reshape(corr_spec, (corr_spec.size / ncorr, ncorr))

    # Rransform the power spectrum back to visibility complex numbers, we assume that phase of complex numbers remains
    # the same throughout AAF.
    corr_spec = vis * (corr_spec / np.reshape(ob_spec, (ob_spec.size / ncorr, ncorr)))
    return corr_spec


def antialias_list(arg_list):
    # This function is just for passing multi arguments to the above funtion corr().
    return antialias_visibilities(*arg_list)


def file_prefix():
    """Return the location of this file, useful for finding the data file"""
    filepath = os.path.split(__file__)
    filepath = os.path.split(filepath[0])
    return filepath[0]


def antialias_ms(msname, tol, outputcolname="DATA_AAF"):
    """
    Apply an anti aliasing filter in a parallel way to a measurement set

    Params:
        msname (str): Name of measurement set
        tol (float): Filter response below this limit will be ignored
        outputcolname (str): Name of column to write corrected visibilities to (will be added to MS if necessary)

    Returns:
        None
    """
    # this function is to implement function corr parallel on the whole MeasurementSet
    # 1. Open MS and read subtable 'DATA',
    #    if necessary create a new column (default "DATA_AAF") to store AAF corrected data.
    t1 = datetime.datetime.now()
    ms = tables.table(msname, readonly=False, ack=False)
    nrows = ms.nrows()
    ini_data = tables.tablecolumn(ms, 'DATA')

    # If there is no column "DATA_AAF", then create one.
    if outputcolname not in ms.colnames():
        coldes = tables.makecoldesc(outputcolname, ms.getcoldesc('DATA'))
        dmname = ms.getdminfo('DATA')
        dmname["NAME"] = 'TiledAAFData'
        ms.addcols(coldes, dminfo=dmname)

    # 2. Calculate function corr()'s two arguments: subband_response and nconv

    # Fixed parameters: Number of channels per subband;  Total number of subbands
    num_channels = 64
    num_subbands = 1024

    # Load filter coefficients, pad with zero
    coeff = np.loadtxt(file_prefix() + '/share/aaf/Coeffs16384Kaiser-quant.dat')
    coeff = np.append(coeff, np.zeros(num_channels * num_subbands - coeff.size))

    # Get filter frequency response by doing FFT on filter coefficients
    frequency_response = np.abs(np.fft.fft(coeff)) ** 2

    # Scaling
    frequency_response = frequency_response / np.sum(frequency_response) * num_channels

    # We only consider aliasing influence from the neighbouring two bands
    subband_response = np.roll(frequency_response, int(1.5 * num_channels))
    subband_response = np.reshape(subband_response[0:3 * num_channels], (3, num_channels))

    # Tolerance, filter response below that is ignored
    # maximum de-convolution length
    nconv = int(math.ceil(math.log(tol, subband_response[2, 1] / subband_response[1, 1])))

    # 3. Do AAF calibration concurrently (parallel)
    num_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cpus - 1)

    # Here itertools and the function antialias_list() are just bridges between pool.map and function corr(), because pool.map
    # is not suitable for function that has multi arguments.
    aafdata = pool.map(antialias_list, itertools.izip(ini_data[0:nrows], itertools.repeat(subband_response), itertools.repeat(nconv)))

    # 4. Write AAF corrected data to MS, usually the size of data is very large (for example 109746*16384*4 in my
    # current MS), to avoid Memory Error, we wrote the data by four steps.
    chunksize = nrows / 4
    start_row = np.array([0, chunksize, chunksize * 2, chunksize * 3])
    end_row = np.array([chunksize, chunksize * 2, chunksize * 3, nrows])
    for parti in range(0, 4):
        ms.putcol(outputcolname, np.array(aafdata[start_row[parti]:end_row[parti]]), startrow=start_row[parti],
                  nrow=end_row[parti] - start_row[parti])
    t2 = datetime.datetime.now()
    print("Total execution time:", (t2 - t1).total_seconds(), "seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apertif Anti-aliasing filter.")
    parser.add_argument("msname", help="Name of Measurement Set")
    parser.add_argument("-t", "--tolerance", help="Filter response below this limit will be ignored", type=float,
                        default=0.00001)
    parser.add_argument("-o", "--output-column", help="Column to output the corrected visibilities to (default DATA_AAF)",
                        default="DATA_AAF")
    args = parser.parse_args()

    antialias_ms(args.msname, args.tolerance, outputcolname=args.output_column)
