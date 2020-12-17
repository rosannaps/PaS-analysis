#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:10:36 2018

@author: rosie
"""

#%% load necessary modules

import neo
import numpy as np
import matplotlib.pyplot as plt
import interpFL
#import time
#import h5py
from detect_peaks import detect_peaks
import grid_strategy as grds
#from matplotlib import gridspec
import dictdiffer
#%%
def load_traces(hyp_filename, dep_filename, cell_chan):
    
    hyp_r = neo.io.AxonIO(filename=hyp_filename)
    block_h = hyp_r.read_block()
    hyp_sweeps = len(block_h.segments)
    dep_r = neo.io.AxonIO(filename=dep_filename)
    block_d = dep_r.read_block()
    dep_sweeps = len(block_d.segments)

    channels = len(block_d.segments[0].analogsignals) # no of channels in dep rec
    channels2  = len(block_h.segments[0].analogsignals) # no of channels in hyp rec
    message1 = None
    message2 = None
    
    hyp_channel_dict = {}
    for ch in range(0, channels2):
        name = block_h.segments[0].analogsignals[ch].name
        if name == '_Ipatch':
            name = 'Ch1'
            block_h.segments[0].analogsignals[ch].name = 'Ch1'
        hyp_channel_dict['AnalogSig%d' % (ch)] = name
    
    dep_channel_dict = {}
    for cd in range(0, channels):
        name = block_d.segments[0].analogsignals[cd].name
        if name == '_Ipatch':
            name = 'Ch1'
            block_d.segments[0].analogsignals[cd].name = 'Ch1'
        dep_channel_dict['AnalogSig%d' %(cd)] = name

    mismatch_chans = []
    for val in hyp_channel_dict.values():
        if val not in dep_channel_dict.values():
            mismatch_chans.append(val)
    for val in dep_channel_dict.values():
        if val not in hyp_channel_dict.values():
            mismatch_chans.append(val)
    
    #compare dictionaries
    if not hyp_channel_dict == dep_channel_dict:
        message1 = 'Warning! Channel mixup'
        print(message1)
        if len(mismatch_chans) <= 0:
            result = dictdiffer.diff(hyp_channel_dict, dep_channel_dict) 
            patched_dep_channel_dict = dictdiffer.patch(result, hyp_channel_dict)
            bl = block_h
            for b in range(0, len(block_d.segments)):
                bl.segments.append(block_d.segments[b])
            sweep_len = len(bl.segments[0].analogsignals[0])
            ch1 = np.ndarray([sweep_len, len(bl.segments)])            
            for key, val in hyp_channel_dict.items():
                if val == 'Ch'+ str(cell_chan): #-1
                    cell = int(key[-1])       
            for i in range(0,len(bl.segments)):
                ch1[:,i] = bl.segments[i].analogsignals[cell].view(np.recarray).reshape(sweep_len)
        else:
            result = dictdiffer.diff(hyp_channel_dict, dep_channel_dict) 
            reslist = list(result)
            bl = block_h
            total_sweeps = len(bl.segments) + len(block_d.segments)
            sweep_len = len(bl.segments[0].analogsignals[0])
            ch1 = np.ndarray([sweep_len, total_sweeps])            
            if 'remove' or 'add' in reslist[0]:
                desired_chan = 'Ch'+str(cell_chan)
            for key in hyp_channel_dict:
                if hyp_channel_dict[key] == desired_chan:
                    cell = int(key[-1])
                for i in range(0,len(bl.segments)):
                    ch1[:,i] = bl.segments[i].analogsignals[cell].view(np.recarray).reshape(sweep_len)
                for key2 in dep_channel_dict:
                    if dep_channel_dict[key2] == desired_chan:
                        cell = int(key2[-1])
                for j, jj in enumerate(range(len(bl.segments), len(bl.segments)+len(block_d.segments))):
                    ch1[:,jj] = block_d.segments[j].analogsignals[cell].view(np.recarray).reshape(sweep_len)
    else:
        message2 = 'Channels match: ok to continue'
        print(message2)
        bl = block_h
        for key, val in hyp_channel_dict.items():
            if val == 'Ch'+ str(cell_chan): #-1
                cell = int(key[-1])       
        for b in range(0, len(block_d.segments)):
            bl.segments.append(block_d.segments[b])
        sweep_len = len(bl.segments[0].analogsignals[cell])
        ch1 = np.ndarray([sweep_len, len(bl.segments)])
        for i in range(0,len(bl.segments)):
            ch1[:,i] = bl.segments[i].analogsignals[cell].view(np.recarray).reshape(sweep_len)
    return ch1, hyp_sweeps, dep_sweeps, sweep_len, bl, message1, message2

def access_resistance(vctpfile, channel,vctp):
    if len(vctp) == 0:
        print('No VCTP file found')
        ResRa = np.nan
        ResRi = np.nan
        pass
    else:
        data=neo.io.AxonIO(vctpfile)
        b1o=data.read_block()
        chan={}
        for ch in range(len(b1o.segments[0].analogsignals)):
            signal=[]
            for s in range(len(b1o.segments)):
                signal.append(b1o.segments[s].analogsignals[ch])
                numb=ch+1
                chan['ch%d'%numb]=[signal,b1o.segments[s].analogsignals[ch].annotations.values()]    #channel is zero-indexed so headstage1 == channel 0 etc etc
        del ch,numb,s,signal
        
        key = 'ch%d' %channel
        mean = np.mean(chan[key][0], axis=0)
        mean = mean.reshape(5000)
        detect_peaks(mean,mpd = 999, edge='both', valley=True) 
        minpeak = np.min(mean)
        avgBL = np.mean(mean[0:1000])
        avgSS = np.mean(mean[np.argmax(mean)-500:np.argmax(mean)-50])
        RaI = avgBL-minpeak
        RiI = avgBL-avgSS
        ResRa = (0.004/(RaI*1e-12))/1000000
        ResRi = (0.004/(RiI*1e-12))/1000000
    return ResRa, ResRi

               
"""Uses median value of first 1000 samples across all sweeps """
def get_baseline(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID):
    if '160817_2' in Cell_ID or '160817_3' in Cell_ID:
        onset = 2469
        hyp_onset = 1000
    elif '160923_7' in Cell_ID or '160923_8' in Cell_ID or '160923_9' in Cell_ID:
        onset = 1000
        hyp_onset = 488
    elif '160701_3'in Cell_ID or '160701_4' in Cell_ID:
        onset = 670
        hyp_onset = 469
    elif Cell_ID == '190926_4':
        onset = 2000
        hyp_onset = 1490
    else:
        onset = np.argmin(np.diff(ch1[0:2050,hyp_sweeps-1], 1))
        r1 = range(990,1010,1)
        r2 = range(1148,1168,1)
        r3 = range(1990,2010,1)
        if onset in r1 or onset in r2 or onset in r3:
            hyp_onset = onset
            pass
        else:
            L = ch1[::4,hyp_sweeps-1]
            dL = np.diff(L,1)
            onset = int(np.argmin(dL[0:600])*4)
            if onset in r1 or onset in r2 or onset in r3:
                hyp_onset = onset
                pass
            else:
                L = ch1[::4,hyp_sweeps-2]
                dL = np.diff(L,1)
                onset = int(np.argmin(dL[0:600])*4)
                if onset in r1 or onset in r2 or onset in r3:
                    hyp_onset = onset
                    pass
                else:
                    L = ch1[::4,hyp_sweeps-3]
                    dL = np.diff(L,1)
                    onset = int(np.argmin(dL[0:600])*4)
                    hyp_onset = onset
#            onset = np.argmin(np.diff(ch1[0:2050,hyp_sweeps-1], 1)) #use minimum of differential to find onset of negative current injection (search within first 100ms of trace to avoid artefacts caused by poor cpn)
    baseline = np.ndarray([len(bl.segments),2])
    for i in range(0, len(bl.segments)):
        baseline[i,1] = np.median(ch1[0:hyp_onset-20,i]) # median value from trace start to 1ms before inj onset
        if i < len(bl.segments)-dep_sweeps:
            baseline[i,0] = (i+1)*-40
        else:
            baseline[i,0] = (i-5)*40
    return baseline, onset, hyp_onset


def vm(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID):
    baseline,onset,hyp_onset = get_baseline(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID)
    Vm = np.mean(baseline[:,1])
    return Vm


def showbaseline(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID):
    baseline, onset, hyp_onset = get_baseline(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID)
    Vm = vm(ch1, bl, dep_sweeps, hyp_sweeps,Cell_ID)
    return


def sag(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID):
    baseline, onset, hyp_onset = get_baseline(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID)
    peak_deflection = np.ndarray([hyp_sweeps, 2])
    for i in range(0,hyp_sweeps):
        peak_deflection[i, 0] = np.argmin(ch1[hyp_onset:hyp_onset+4000,i])+hyp_onset #search xxms window after onset to select local minimum of deflection
        peak_deflection[i, 1] = np.min(ch1[hyp_onset:hyp_onset+4000,i])
    offset = hyp_onset+20000
    steady_state = np.ndarray([hyp_sweeps,1])
    for i in range(0,hyp_sweeps):
        steady_state[i,0] = np.median(ch1[offset-2000: offset-1, i])
    delta_peak_deflection = peak_deflection[:,1] - baseline[0:hyp_sweeps,1]
    delta_ss_deflection = steady_state[:,0] - baseline[0:hyp_sweeps, 1]
    sag_ratio = delta_ss_deflection/delta_peak_deflection
    return sag_ratio, peak_deflection


def membrane_constant(ch1,hyp_sweeps,hyp_onset):
    offset = hyp_onset+20000
    mc = np.ndarray([hyp_sweeps,3])
    for i in range(0,hyp_sweeps):
        I = (-40*(i+1))*1e-12
        bl = np.median(ch1[0:hyp_onset-20,i])
        ss = np.median(ch1[offset-2000:offset-1,i])
        swp = list(ch1[:,i])
        Vdiff = bl-ss
        v65 = Vdiff*0.63
        V65 = bl-v65
        res = list(filter(lambda ii: ii < V65, swp))[0]
        tau65 = swp.index(res)
        R = (Vdiff/1000)/-I     
        tc = tau65 - hyp_onset
        mc[i,0] = tc*0.05
        mc[i,1] = R*1e-6
        mc[i,2] = tc*5e-5/R
    mc[:,2] = mc[:,2]/1e-12
    return mc    


def APpanelplot(first_spike, inj):
    win = len(inj) - first_spike
    grid = grds.GridStrategy.get_grid(win)
    ax_spec = grds.get_gridspec(grid)
    return ax_spec


def apcount(ch1, dep_sweeps, hyp_sweeps,dep_file, cell_chan, deps):
    max_current_injection = dep_sweeps*40
    inj = range(0, max_current_injection, 40)
  
# get max no of spikes, loading subsequent dep files if necessary&available
    max_spikes = 0
    for i, j in enumerate(range(hyp_sweeps, len(ch1[0]))):
        pks = detect_peaks(ch1[:,j], mph=0,mpd=50)
        if len(pks)> max_spikes:
            max_spikes = len(pks)

    if max_spikes < 10 and len(deps)> 1:
            trace = deps[1]
            dep2_filename = dep_file[:-7]+trace+'.abf'
            info = neo.AxonIO(dep2_filename)._axon_info
            dep_r2 = neo.io.AxonIO(filename=dep2_filename)
            block_d2 = dep_r2.read_block()
            channels = len(block_d2.segments[0].analogsignals) # no of channels in dep rec
            dep2_channel_dict = {}
            for cd in range(0, channels):
                name = block_d2.segments[0].analogsignals[cd].name
                if name == '_Ipatch':
                    name = 'Ch1'
                    block_d2.segments[0].analogsignals[cd].name = 'Ch1'
                dep2_channel_dict['AnalogSig%d' %(cd)] = name
            for key, val in dep2_channel_dict.items():
                if val == 'Ch'+ str(cell_chan): #-1
                    cell = int(key[-1])       
            for seg in block_d2.segments:
                sweep = seg.analogsignals[cell].view(np.recarray)
                ch1 = np.append(ch1,sweep, axis=1)
            injection = int(info['dictEpochInfoPerDAC'][cell_chan-1][1]['fEpochInitLevel'])
            new_sweeps = len(block_d2.segments)
            dep_sweeps = dep_sweeps + new_sweeps
            inj=list(inj)
            new_inj = list(range(injection, injection+new_sweeps*40, 40))
            inj.extend(new_inj)
            for i, j in enumerate(range(hyp_sweeps, len(ch1[0]))):
                pks = detect_peaks(ch1[:,j], mph=0, mpd=50)
                if len(pks)> max_spikes:
                    max_spikes = len(pks)
            if max_spikes < 10 and len(deps) > 2:
                    trace2 = deps[2]
                    dep3_filename = dep_file[:-7]+trace2+'.abf'
                    info2 = neo.AxonIO(dep3_filename)._axon_info
                    dep_r3 = neo.io.AxonIO(filename=dep3_filename)
                    block_d3 = dep_r3.read_block()
                    channels = len(block_d3.segments[0].analogsignals) # no of channels in dep rec
                    dep3_channel_dict = {}
                    for cd in range(0, channels):
                        name = block_d3.segments[0].analogsignals[cd].name
                        if name == '_Ipatch':
                            name = 'Ch1'
                    block_d3.segments[0].analogsignals[cd].name = 'Ch1'
                    dep3_channel_dict['AnalogSig%d' %(cd)] = name
                    for key, val in dep2_channel_dict.items():
                        if val == 'Ch'+ str(cell_chan): #-1
                            cell = int(key[-1])       
                    for seg in block_d3.segments:
                        sweep = seg.analogsignals[cell].view(np.recarray)
                        ch1 = np.append(ch1,sweep, axis=1)
                    injection2 = int(info2['dictEpochInfoPerDAC'][cell_chan-1][1]['fEpochInitLevel'])
                    new_sweeps2 = len(block_d3.segments)
                    new_inj2 = list(range(injection2, injection2+new_sweeps2*40, 40))
                    dep_sweeps = dep_sweeps + new_sweeps2
                    inj.extend(new_inj2)
                    for i, j in enumerate(range(hyp_sweeps, len(ch1[0]))):
                        pks = detect_peaks(ch1[:,j], mph=0,mpd=50)
                        if len(pks)> max_spikes:
                            max_spikes = len(pks)
                    if max_spikes == 0:
                        print('No APs found')
            else:
                pass
    else:
        pass
            
    spike_counts = np.ndarray([len(inj),2])
    peaks = np.empty([len(inj), max_spikes, 3])
    peaks.fill(np.nan)
    for i, j in enumerate(range(hyp_sweeps, len(ch1[0]))):
        pks = detect_peaks(ch1[:,j], mph=0,mpd=50)
        peaks[i,0:len(pks), 1] = pks
        peaks[i,0:len(pks), 0] = inj[i]
        peaks[i,0:len(pks), 2] = ch1[pks,j]
# number of spikes in each step
    for i, j in enumerate(inj):
        spike_counts[i,0] = j
        spike_counts[i,1] = (np.sum(np.isfinite(peaks[i,:,:])))/3
# find first step with spikes
#    plt.figure()
#    for i in range(0, len(ch1[0])):
#        plt.plot(ch1[:,i], lw=0.2)
    spikes = np.where(np.isfinite(peaks))
    first_spike = spikes[0][0]
    IOfit = np.polyfit(spike_counts[first_spike:,0], spike_counts[first_spike:,1],1)
    
    IO_slope = IOfit[0]
    return inj, spike_counts, max_spikes, peaks, first_spike, IO_slope, ch1, dep_sweeps


def rheobase(inj, first_spike):
    Rheobase = inj[first_spike]   
    return Rheobase


def check_single_currinj(curinj, dep_sweeps, peaks, inj, ch1, hyp_sweeps, max_spikes):
    max_current_injection = (dep_sweeps)*40
    inj = range(0,max_current_injection,40)
    peaks = np.empty([1, max_spikes, 2])
    peaks.fill(np.nan)
    pks = detect_peaks(ch1[:,hyp_sweeps+inj.index(curinj)], mph=0,mpd=50)
    peaks[0,0:len(pks), 0] = pks
    peaks[0,0:len(pks), 1] = ch1[pks,hyp_sweeps+inj.index(curinj)]
    if not np.any(np.isin(inj, curinj)):
        print('Current injection does not exist!')
    else:
        fig = plt.figure()
        plt.plot(ch1[:,hyp_sweeps+inj.index(curinj)], c='k', lw=0.5)
        plt.scatter(peaks[0, :, 0], peaks[0, :, 1],
                    c='r', marker='+')
    return fig


def IV_curve(hyp_sweeps, first_spike, peak_deflection, ch1, Vm, onset):
#hits error if cell spikes during first dep sweep (ie when no current injection)
    no_traces = hyp_sweeps + first_spike   
    IV_outcome = np.ndarray([no_traces, 2])
    for i,k in enumerate(range(hyp_sweeps,0, -1)):
        IV_outcome[i,0] = k*-40
        IV_outcome[i,1] = peak_deflection[k-1,1]
    if first_spike == 0:
        pass
    else:
        IV_outcome[hyp_sweeps,1] = Vm
        IV_outcome[hyp_sweeps,0] = 0
        for i,k in enumerate(range(hyp_sweeps+1,len(IV_outcome))):
            IV_outcome[k,0] = (i+1)*40
            IV_outcome[k,1] = np.max(ch1[onset:onset+1500,k])
    subTHiv_fit = np.polyfit(IV_outcome[:,0], IV_outcome[:,1], 1)
    IR = subTHiv_fit*1000
    return subTHiv_fit, IR

def threshold(ch1, hyp_sweeps, inj, spike_counts, peaks, dep_sweeps, first_spike, max_spikes):  
    TH = np.empty([len(inj), max_spikes, 4])
    TH.fill(np.nan)
    d1 = np.diff(ch1[:,hyp_sweeps:],1, axis=0)
#need to rule out artefact sometimes present from CpN - shift TH up to 2
    d1peaks = np.empty([len(inj), max_spikes, 3])
    d1peaks.fill(np.nan)
###find maximal differential for each AP
    for i in range(0, len(d1[0])):
        d1pks = detect_peaks(d1[:,i], mph=2, mpd=5)
        if len(d1pks) == 0:
            d1peaks[i,:, 0] = inj[i]
        if len(d1pks)>1 and d1[d1pks[0],i] < d1[d1pks[1],i]/2:#checks for peak caused by stim on artefact and removes if finds first peak is less than half size of second. Changed first argument to greater than 1 (previously 0) on 180517 
            d1pks = np.delete(d1pks,0)
        if len(d1pks) > spike_counts[i,1]:
            d1peaks[i,0:int(spike_counts[i,1]), 1] = d1pks[0:int(spike_counts[i,1])]
            d1peaks[i,0:int(spike_counts[i,1]), 2] = d1[d1pks[0:int(spike_counts[i,1])],i]
            d1peaks[i,0:len(d1pks), 0] = inj[i]
        else:    
            d1peaks[i,0:len(d1pks), 1] = d1pks
            d1peaks[i,0:len(d1pks), 2] = d1[d1pks,i]
            d1peaks[i,0:len(d1pks), 0] = inj[i]  
###Now find TH by using 5% max diff for each individual AP
    TH[:,:,0] = d1peaks[:,:,0]
    TH[:,:,1] = d1peaks[:,:,2]*0.05 # 5% of max
    for i in range(0, dep_sweeps):
        for j in range(0,len(peaks[1])):#just do first 10 APs
            if np.isnan(peaks[i,j,1]):
                pass
            else:
                TH[i,j,2] = peaks[i,j,1]-50 + np.argmax(d1[int(peaks[i,j,1]-50):int(peaks[i,j,1]), i]>TH[i,j,1]) +1   # replaced 100 with 50 as test - kept change as fixed problem - 190508.
                TH[i,j,3] = ch1[int(TH[i,j,2]),i+hyp_sweeps]
    return TH, d1


def rel_firing_freq(first_spike, spike_counts): 
    #relative firing freq is taken two current steps above rheobase.
    #However, if cell becomes inactive before this then either is taken,
    # a) the last value of spike counts (i.e. last current injection) 
    # b) if two steps above rheoabse the firing freq is lower than either 
    #one step above or rheobase itself then the max value of either rheobase 
    #or rheobase+1 is taken.
    if first_spike+2 >= len(spike_counts):
        Firing_freq = int(spike_counts[-1,1])   
    elif spike_counts[first_spike+2,1] < spike_counts[first_spike,1] or spike_counts[first_spike+2,1] < spike_counts[first_spike+1,1]:
        Firing_freq = int(max(spike_counts[first_spike+1,1], spike_counts[first_spike, 1]))
    else:
        Firing_freq = int(spike_counts[first_spike+2,1])
    return Firing_freq


def max_firing_freq(spike_counts):
    max_freq = max(spike_counts[:,1])
    return max_freq


def slopes(TH, first_spike, d1, peaks):   
    AP1_max_deriv = max(d1[int(TH[first_spike][0][2]):int(peaks[first_spike][0][1]),first_spike])
    AP1_min_deriv = min(d1[int(TH[first_spike][0][2]):int(TH[first_spike][0][2]+100),first_spike])
    slope_ratio = np.abs(AP1_max_deriv/AP1_min_deriv)
    return AP1_max_deriv, AP1_min_deriv, slope_ratio


def mahp(dep_sweeps, ch1, hyp_sweeps, onset):   
    offset = onset+20000
    mAHP = np.empty([dep_sweeps,3])
    for i in range(0,dep_sweeps):
        mAHP[i,0] = np.min(ch1[offset:offset+4000, i+hyp_sweeps]) #first column is minimum value after stimoff
        mAHP[i,1] = np.median(ch1[0:onset,i+hyp_sweeps])   #Vm from start of corresponding trace
    mAHP[:,2] = mAHP[:,1] - mAHP[:,0] #difference = mAHP
    return mAHP


def latency(TH, inj, onset):    
    Latency = np.ndarray([len(inj),2])
    for i, j in enumerate(inj):
        Latency[i,1] = (TH[i,0,2] - onset)/20
        Latency[i,0] = j
    return Latency


def ap_height(peaks, first_spike, TH):        
    AP_height = peaks[first_spike][0][2] - TH[first_spike][0][3]    
    return AP_height


def intervals(TH, spike_counts, max_spikes):   
    if max_spikes <= 1:
        AP12int = np.nan
        AP910int = np.nan
        adaptation = np.nan
        ISI = np.nan
    elif max_spikes <=5:
        AP910int = np.nan
        adaptation = np.nan
        ISI = np.copy(TH[np.nonzero(spike_counts[:,1])[0][0]:,:,0:3])
        ISI[:,:,1].fill(np.nan)
        for i in range(len(ISI)):
            for j in range(0, (np.count_nonzero(~np.isnan(ISI[i,:,2]))-1)):
                ISI[i, j, 1] = (ISI[i, j+1, 2] - ISI[i, j, 2])/20.0
                usetrace = np.argmax(spike_counts[:,1])
                AP12int = (TH[usetrace,1,2] - TH[usetrace,0,2])/20
    else:
        ISI = np.copy(TH[np.nonzero(spike_counts[:,1])[0][0]:,:,0:3])
        ISI[:,:,1].fill(np.nan)
        for i in range(len(ISI)):
            for j in range(0, (np.count_nonzero(~np.isnan(ISI[i,:,2]))-1)):
                ISI[i, j, 1] = (ISI[i, j+1, 2] - ISI[i, j, 2])/20.0
        if np.nanmax(spike_counts[:,1]) < 10:
            usetrace = np.argmax(spike_counts[:,1])
            AP12int = (TH[usetrace,1,2] - TH[usetrace,0,2])/20
            AP23int = (TH[usetrace,2,2] - TH[usetrace, 1,2])/20
            AP910int = (TH[usetrace,-1,2] - TH[usetrace,-2,2])/20
        else:
            tenAPind = np.where(spike_counts[:,1] >=10)
            AP12int = (TH[tenAPind,1,2] - TH[tenAPind,0,2])/20  # in ms
            AP23int = (TH[tenAPind,2,2] - TH[tenAPind,1,2])/20  # in ms
            AP910int = (TH[tenAPind,9,2] - TH[tenAPind,8,2])/20 # in ms
        if not isinstance(AP910int, np.ndarray) or isinstance(AP23int,np.ndarray):
            adaptation = AP910int/AP23int
        else:
            adaptation = AP910int[0][0]/AP23int[0][0]
        
    if type(AP12int) == np.ndarray:
        AP12 = AP12int[0][0].item()
    elif np.isnan(AP12int):
        AP12 = np.nan
    else:
        AP12 = AP12int
              
    if type(AP910int) == np.ndarray:
        AP910 = AP910int[0][0].item()
    elif np.isnan(AP910int):
        AP910 = np.nan
    else:
        AP910 = AP910int
                
    if type(adaptation) == np.ndarray:
        adaptn = adaptation[0][0].item()
    elif np.isnan(adaptation):
        adaptn = np.nan
    else:
        adaptn = adaptation
        
    if type(ISI) == np.ndarray:
        if np.nanmin(ISI[:,:,1]) <= 3:
            ISI_res = np.nanmin(np.ma.masked_less(ISI,3)[:,:,1])
        else: 
            ISI_res = np.nanmin(ISI[:,:,1])
    elif np.isnan(ISI):
        ISI_res = np.nan
    else:
        ISI_res = ISI
        
    return AP12, AP910, adaptn, ISI_res


def cut_ap(TH, ch1, hyp_sweeps, spike_counts, first_spike, peaks, sweep=None, spike=1):   #enter spike number as you would count (ie not 0-indexed)
    if sweep is None:
        sweep = first_spike
    start_cut = int(TH[sweep][spike-1][2])  # correct spike for 0-indexing
    if spike+1 > spike_counts[sweep][1]:  # if desired spike is last in sweep cut 2ms after
        end_cut = int(peaks[sweep][spike-1][1])+40  #edited on 190319 to change end cut to 2ms after peak instead of onset
    else:
        end_cut = int(TH[sweep][spike][2])  #else cut at TH of next spike
    cut_AP = ch1[start_cut:end_cut,hyp_sweeps+sweep]
    return cut_AP


def halfwidth(first_spike, TH, ch1, hyp_sweeps, spike_counts, peaks, sweep=None, spike=1):
    AP = cut_ap(TH, ch1, hyp_sweeps, spike_counts, first_spike,peaks, sweep=None, spike=1)
    AP_amp = max(AP) - AP[0]
    half_amp = AP_amp/2 + AP[0]
    t10 = AP_amp*0.1+AP[0]
    t90 = AP_amp*0.9+AP[0]
    fit_start = interpFL.findLevels(AP, t10, mode='rising')
    fit_end = interpFL.findLevels(AP, t90, mode='rising')
    #first fit rising slope using 10-90% max amp (don't forget to add minimum AP (start) value)
    x = range(len(AP))
    x1 = x[fit_start[0][0]+1:fit_end[0][0]+1]
    fit1 = np.polyfit(x1, AP[x1], 1)
    hw1 = (half_amp-fit1[1])/fit1[0]    
    #fit falling phase: look at fitting 10-90% eg of slope - max to min is not satisfactory
    fall_amp = max(AP) - min(AP)
    f10 = fall_amp*0.1+min(AP)
    f90 = fall_amp*0.9+min(AP)
    fit2_start = interpFL.findLevels(AP,f90, mode='falling')
    fit2_end = interpFL.findLevels(AP, f10, mode='falling')
    x2 = x[fit2_start[0][0]+1:fit2_end[0][0]+1]
    fit2 = np.polyfit(x2, AP[x2], 1)
    hw2 = (half_amp-fit2[1])/fit2[0]
    HW = (hw2-hw1)/20 #conversion to ms
    return HW


def ahp(TH, ch1, hyp_sweeps, spike_counts, first_spike, peaks, sweep=None, spike=1):  
    AHP_AP = cut_ap(TH, ch1, hyp_sweeps, spike_counts, first_spike, peaks,sweep=first_spike, spike=1)
    AHP = AHP_AP[0] - min(AHP_AP[np.argmax(AHP_AP):-1])
    return AHP
