from casacore import tables
import numpy as np
import math
import multiprocessing
import itertools
import datetime
def corr(ini_ob_spec,Re_sb,nconv):  
    #this is the function that does the AAF calibration
    #Arguments
    #ini_ob_spec:  input spectrum, which is supposed from raw MS(visibility), datatype complex numbers, shape(Nsb*64,correlator numbers)
    #Re_sb: filter frequency response,see annotation in the main function
    #nconv: de-convolution length. see annotation in the main function
    #Return value:
    #corr_spec: AAF corrected spectrum, the same shape and datatype as ini_ob_spec
    
    
    
    #1.Preparation before implementation AAF
    
    #caculate spectrum's subbands numbers, correlator numbers
    ncorr=ini_ob_spec.shape[1]
    Nch=64
    Nsb=ini_ob_spec.shape[0]/64
    
    #calculate the power spectrum, reshape spectrum
    ob_spec=abs(ini_ob_spec) 
    ob_spec=np.reshape(ob_spec,(Nsb,Nch,ncorr))  
        
    #estimate missing subbands
    #find missing subbands and create an array to record the location of missing subbands
    sumspec=np.sum(ob_spec,axis=1)
    zeros=np.where(sumspec ==0)
    flag=np.zeros(ob_spec.shape)
    flag[zeros[0],:,zeros[1]]=1
    #use linear interpolation to estimate the missing subbands
    ob_spec1=list((np.swapaxes(ob_spec,0,2)).reshape(ob_spec.shape[1]*ob_spec.shape[2],ob_spec.shape[0]))
    ob_spec2=np.swapaxes(np.array([np.interp(np.arange(0,ob_spec.shape[0],1),np.where(x >0)[0],x[np.where(x >0)]) for x in ob_spec1]).reshape(ob_spec.shape[2],ob_spec.shape[1],ob_spec.shape[0]),0,2)
    ob_spec=ob_spec2.copy()
    

    #2.begin AAF calibration:  
    
    #This algorithm also does bandpass correction!!!!!!
    #In case the input spectrum has already been bandpass corrected,  de-annotate the next row to undoes this effect. 
    #ob_spec=np.swapaxes((np.swapaxes(ob_spec,1,2)/Re_sb[1,:]),1,2)
    
    #create and initialize a new array to store AAF corrected spectrum
    corr_spec=np.empty_like(ob_spec)
    corr_spec[:]=np.nan
    
    #begin with central channel, suppose the central channel of each subband is not influenced by aliasing effect
    corr_spec[:,Nch/2,:]=ob_spec[:,Nch/2,:]/Re_sb[1,Nch/2]

    #from central channel downwards,ignore response of previous subband(for a certain subband, channel1 to channel 31 was mostly influenced by next subband's channel1 to channel31)
    for chidx in np.arange(Nch/2-1,0,-1):
        ratio=-1.*Re_sb[2,chidx]/Re_sb[1,chidx]
        f_corr=ratio**(np.arange(nconv-1,-1,-1))/Re_sb[1,chidx]
        for corri in np.arange(0,ncorr,1):
            corr_spec[:,chidx,corri]=(np.convolve(ob_spec[:,chidx,corri],f_corr))[nconv-1:] 
            #compensate for missing smaple
        if chidx < nconv:
            f_corr=f_corr*ratio
            #estimate of missing data, use neighbouring channel as initial estimate
            ini_spec=corr_spec[:,chidx+1,:]
            dmissing=np.ones(ncorr)
            for corri in np.arange(0,ncorr,1): #this for-loop is somehow unavoidable, because np.linalg.lstsq only accept one-dimensional array while our data array is too deep.
                dmissing[corri]=(np.linalg.lstsq(np.transpose(np.mat(f_corr)),np.transpose(np.mat(ini_spec[Nsb-nconv:,corri]-corr_spec[Nsb-nconv:,chidx,corri]))))[0][0,0]
                corr_spec[Nsb-nconv:,chidx,corri]=corr_spec[Nsb-nconv:,chidx,corri]+np.dot(f_corr,dmissing[corri])
    #from central channel upwards,ignore response of next subband (for a certain subband, channel33 to channel 63 were mostly influenced by previous subband's channel33 to channel63)
    for chidx in np.arange(Nch/2+1,Nch,1):
        ratio=-1.*Re_sb[0,chidx]/Re_sb[1,chidx]
        f_corr=ratio**(np.arange(0,nconv,1))/Re_sb[1,chidx]
        for corri in np.arange(0,ncorr,1):
            corr_spec[:,chidx,corri]=(np.convolve(ob_spec[:,chidx,corri],f_corr))[0:Nsb]
        if chidx > Nch-nconv-1:
            f_corr=f_corr*ratio
            #estimate of missing data,use neighbouring channel as initial estimate
            ini_spec=corr_spec[:,chidx-1,:]
            for corri in np.arange(0,ncorr,1):
                dmissing[corri]=(np.linalg.lstsq(np.transpose(np.mat(f_corr)),np.transpose(np.mat(ini_spec[0:nconv,corri]-corr_spec[0:nconv,chidx,corri]))))[0][0,0]
                corr_spec[0:nconv,chidx,corri]=corr_spec[0:nconv,chidx,corri]+np.dot(np.reshape(f_corr,(1,f_corr.size)),dmissing[corri])
    
    #dealing with the first channels, since eventually we'll just ignore or flag first channels, so it's okay to annotate this block, might save some time
    nedge=3
    chidx=0
    ratio=-1.*Re_sb[2,chidx]/Re_sb[1,chidx]
    f_corr=ratio**(np.arange(Nsb-1,-1,-1))/Re_sb[1,chidx]
    #estimate missing data,use average of first and last channel as initial estimate
    ini_spec=(corr_spec[:,1,:]+np.roll(corr_spec[:,Nch-1,:],1,axis=0))/2.
    ini_spec[0,:]=corr_spec[0,1,:]
    dmissing=np.ones(ncorr)
    for corri in np.arange(0,ncorr,1):
        corr_spec[:,chidx,corri]=(np.convolve(ob_spec[:,chidx,corri],f_corr))[np.size(f_corr)-1:]
    f_corr=ratio*f_corr
    for corri in np.arange(0,ncorr,1):
        dmissing[corri]=(np.linalg.lstsq(np.transpose(np.mat(f_corr[nedge:Nsb-nedge])),np.transpose(np.mat(ini_spec[nedge:Nsb-nedge,corri]-corr_spec[nedge:Nsb-nedge,chidx,corri]))))[0][0,0]
        corr_spec[:,chidx,corri]=corr_spec[:,chidx,corri]+np.dot(f_corr,dmissing[corri])
    
    
    
    #3.flagging and reshaping AAF corrected spectrum,transform power spectrum back to visibility complex numbers

    #flag the missing subbands and negative values
    corr_spec[np.where(corr_spec <0)]=np.nan
    corr_spec[np.where(flag==1)]=np.nan
    #reshape spectrum
    corr_spec=np.reshape(corr_spec,(corr_spec.size/ncorr,ncorr))
    #transform the power spectrum back to visibility complex numbers, we assume that phase of complex numbers remains the same throughout AAF.
    corr_spec=ini_ob_spec*(corr_spec/np.reshape(ob_spec,(ob_spec.size/ncorr,ncorr)))
    return corr_spec



def conver(a_b):
    #this function is just for passing multi arguments to the above funtion corr().
    return corr(*a_b)




def MS_corr(msname,tol):
    #this function is to implement function corr parallel on the whole MeasurementSET
    #1.open MS and read subtable 'DATA', create a new subtable "DATA_AAF" to store AAF corrected data.
    t1=datetime.datetime.now()
    ms=tables.table(msname,readonly=False)
    nrows=ms.nrows()
    ini_data=tables.tablecolumn(ms,'DATA')
    
    #if there is no subtable "DATA_AAF", then create one.
    if "DATA_AAF" not in ms.colnames():
        coldes=tables.makecoldesc('DATA_AAF',ms.getcoldesc('DATA'))
        dmname=ms.getdminfo('DATA')
        dmname["NAME"]='TiledAAFData'
        ms.addcols(coldes,dminfo=dmname)
    
    
    
    #2.calculate function corr()'s two arguments:Re_sb and nconv
    
    #fixed parameters: Number of channels per subband;  Total number of subbands
    Nch=64
    Nsb_total=1024
    
    #load filter coefficients, pad with zero
    coeff=np.loadtxt('Coeffs16384Kaiser-quant.dat')
    coeff=np.append(coeff,np.zeros(Nch*Nsb_total-coeff.size))
    #get filter frequency response by doing FFT on filter coefficients
    Re_fren=np.abs(np.fft.fft(coeff))**2
    #scaling
    Re_fren=Re_fren/np.sum(Re_fren)*Nch
    #We only conside aliasing influence from the neighbouring two bands
    Re_sb=np.roll(Re_fren,int(1.5*Nch))
    Re_sb=np.reshape(Re_sb[0:3*Nch],(3,Nch))
    
    #tolerance,filter response below that is ignored
    #maximum de-convolution length
    nconv=int(math.ceil(math.log(tol,Re_sb[2,1]/Re_sb[1,1])))
    
    
    #3.do AAF calibration concurrently (parallel)
    Ncpu=multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=Ncpu-1)
    #here itertools and the function conver() are just bridges between pool.map and function corr(), because pool.map is not suitable for function that has multi arguments.
    aafdata=pool.map(conver,itertools.izip(ini_data[0:nrows],itertools.repeat(Re_sb),itertools.repeat(nconv)))
    
    
    
    #4.write AAF corrected data to MS, usually the size of data is very large(for example 109746*16384*4 in my current MS), to avoid Memory Error, we wrote the data by four steps.
    l=nrows/4
    stx=np.array([0,l,l*2,l*3])
    endx=np.array([l,l*2,l*3,nrows])
    for parti in xrange(0,4):
        ms.putcol('DATA_AAF',np.array(aafdata[stx[parti]:endx[parti]]),startrow=stx[parti],nrow=endx[parti]-stx[parti])
    t2=datetime.datetime.now()
    print "\ntotal execution time:",(t2-t1).total_seconds(),"seconds"
if __name__=='__main__':
    print ""
