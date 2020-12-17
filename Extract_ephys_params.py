#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:14:04 2020

@author: rosie
"""
#import required modules

import new_char_functions_v6 as ncf
import time
import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import gc
#%% 
# =============================================================================
# initiate dict for kept cells and failed cells (i.e. ones that run into errors)
# =============================================================================
cell_dic2={}
failed_dic2={}
#%% 
# =============================================================================
# read in indexing spreadsheet and get list of cellIDs
# =============================================================================
book = xlrd.open_workbook('/Volumes/TOSHIBA EXT/PaS Electrophysiology/Recording_indexing2 2.xlsx')
sheet = book.sheet_by_name('Sheet1')
#create list of cell names to search for CellID and get row of cellID (=entry)
cell_names = []
for r in range(sheet.nrows-1):
    cell_names.append(str(sheet.cell_value(r+1,0))+"_"+ str(int(sheet.cell_value(r+1,1))))
#%% 
# =============================================================================
# loop over each cell running functions from characterisation script
# =============================================================================
for num,name in enumerate(cell_names):
    try:
        #get relevant filenames and recording channel
        Cell_ID=name
        entry = cell_names.index(name) +1
        
        hyp = sheet.cell(entry,5).value
        hyps = hyp.split(',')
        hyp_file = str(sheet.cell_value(entry,4))+str(hyps[0])+'.abf'
        
        dep = sheet.cell(entry,6).value
        deps = dep.split(',')
        dep_file = str(sheet.cell_value(entry, 4))+str(deps[0])+'.abf'
        
        vctp = sheet.cell(entry,9).value
        vctps = vctp.split(',')
        vctpfile = str(sheet.cell_value(entry,4)+str(vctps[0])+'.abf')
        channel = int(sheet.cell(entry, 11).value)
        
        cell_type = sheet.cell_type(entry,17)
        if cell_type == xlrd.XL_CELL_EMPTY:
            holding_current = np.nan
        else:
            holding_current = sheet.cell_value(entry,17)        
        
        #run functions
        start = time.time()
        ch1, hyp_sweeps, dep_sweeps, sweep_len, bl, message1, message2 = ncf.load_traces(hyp_file, dep_file, channel)    
        Ra, Ri = ncf.access_resistance(vctpfile, channel,vctp)
        baseline, onset, hyp_onset = ncf.get_baseline(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID)
        Vm = ncf.vm(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID)
        sag_ratio, peak_deflection = ncf.sag(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID)
        mc = ncf.membrane_constant(ch1, hyp_sweeps, hyp_onset)
        inj, spike_counts, max_spikes, peaks, first_spike, IO_slope, ch1, dep_sweeps = ncf.apcount(ch1, dep_sweeps, hyp_sweeps, dep_file, channel, deps)        
        ax_spec = ncf.APpanelplot(first_spike, inj)  
        Rheobase = ncf.rheobase(inj, first_spike)    
        subTHiv_fit, IR = ncf.IV_curve(hyp_sweeps, first_spike, peak_deflection, ch1, Vm, onset)    
        TH, d1 = ncf.threshold(ch1, hyp_sweeps, inj, spike_counts, peaks, dep_sweeps, first_spike, max_spikes)  
        Firing_freq = ncf.rel_firing_freq(first_spike, spike_counts)
        max_firing_freq = ncf.max_firing_freq(spike_counts)  
        AP1_max_deriv, AP1_min_deriv, slope_ratio = ncf.slopes(TH, first_spike, d1, peaks)  
        mAHP = ncf.mahp(dep_sweeps, ch1, hyp_sweeps, onset)   
        Latency = ncf.latency(TH, inj, onset)  
        AP_height = ncf.ap_height(peaks, first_spike, TH)  
        AP12, AP910, adaptn, ISI_res = ncf.intervals(TH, spike_counts, max_spikes) 
        cut_AP = ncf.cut_ap(TH, ch1, hyp_sweeps, spike_counts, first_spike,peaks,sweep=None, spike=1)   
        HW = ncf.halfwidth(first_spike, TH, ch1, hyp_sweeps, spike_counts,peaks,sweep=None, spike=1)  
        AHP = ncf.ahp(TH, ch1, hyp_sweeps, spike_counts, first_spike,peaks,sweep=None, spike=1)
        plt.close('all')            
                
        notes = ''
        count = len(cell_dic2.keys())
        new_entry = count+1
        #enter values into dic
        cell_dic2[new_entry] = {'CellID': Cell_ID, 'Vm':Vm, 'Sag Ratio':sag_ratio[-1],
                'Membrane time constant': mc[3,0], 'Membrane capacitance': mc[3,2], 
                'Input Resistance': IR[0], 'mAHP':mAHP[-1][2],
                'Latency': Latency[np.where(~np.isnan(Latency[:,1]))[0][0],1], 
                'AP1-AP2int':AP12, 'AP9-AP10int':AP910, 'Adaptation': adaptn, 
                'AP height': AP_height, 'Max dV/dt': AP1_max_deriv,
                'Min dV/dt': AP1_min_deriv, 'dV/dt ratio': slope_ratio, 
                'TH': TH[first_spike, 0,3], 'AHP': AHP, 'Halfwidth': HW, 
                'Rheobase': Rheobase, 'Rel. firing freq': Firing_freq, 
                'Max firing freq': max_firing_freq, 'minISI': ISI_res, 'Rin':Ri,
                'Rser':Ra,'Holding current': holding_current, 'Notes': notes}
        gc.collect()
    #cells that return error from any function are entered here    
    except:
        count2 = len(failed_dic2.keys())
        failed_entry = count2+1
        failed_dic2[failed_entry] = Cell_ID
        pass
    

#%% #
# =============================================================================
# manual addition of cells that may have returned error but can still be added
# cases include where <10 spikes and interval function produces error
# or earlier version of VCTP protocol run so onset/offset/sweep length are diff
# Usually prefer in this case to run version of script that doesn't suppress
# plots so they can be checked over.     
# =============================================================================
#
# import functions script version with plots not suppressed
import new_char_functions_output_plots_v4 as ncfO

CellID = input('Enter the cell ID:')

book = xlrd.open_workbook('/Volumes/TOSHIBA EXT/PaS Electrophysiology/Recording_indexing2 2.xlsx')
#book = xlrd.open_workbook('/Volumes/AG-Schmitz-2/Rosie/PaS Connectivity/Recording_indexing2 2.xlsx')

sheet = book.sheet_by_name('Sheet1')

#create list of cell names to search for CellID and get row of cellID (=entry)
cell_names = []
for r in range(sheet.nrows-1):
    cell_names.append(str(sheet.cell_value(r+1,0))+"_"+ str(int(sheet.cell_value(r+1,1))))
entry = cell_names.index(CellID) + 1

# check how many dep and hyp traces were collected, as default load first - can add a checking/changing option later

hyp = sheet.cell(entry,5).value
hyps = hyp.split(',')
hyp_file = str(sheet.cell_value(entry,4))+str(hyps[0])+'.abf'

dep = sheet.cell(entry,6).value
deps = dep.split(',')
dep_file = str(sheet.cell_value(entry, 4))+str(deps[0])+'.abf'

vctp = sheet.cell(entry,9).value
vctps = vctp.split(',')
vctpfile = str(sheet.cell_value(entry,4)+str(vctps[0])+'.abf')
channel = int(sheet.cell(entry, 11).value)

cell_type = sheet.cell_type(entry,17)
if cell_type == xlrd.XL_CELL_EMPTY:  
    holding_current = np.nan
else:
    holding_current = sheet.cell_value(entry,17)        
Cell_ID=CellID
#Bits of useful code for certain errors, if first function needs manually running
#
cell_chan = channel
dep_filename = dep_file
hyp_filename = hyp_file
##           
#
#Ri = np.nan
#Ra = np.nan

#%% 
# =============================================================================
# run functions
# =============================================================================
start = time.time()
ch1, hyp_sweeps, dep_sweeps, sweep_len, bl, message1, message2 = ncfO.load_traces(hyp_file, dep_file, channel)
Ra, Ri = ncfO.access_resistance(vctpfile, channel,vctp)
baseline, onset, hyp_onset = ncfO.get_baseline(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID)
Vm = ncfO.vm(ch1, bl, dep_sweeps, hyp_sweeps,Cell_ID)
sag_ratio, peak_deflection = ncfO.sag(ch1, bl, dep_sweeps, hyp_sweeps, Cell_ID)
mc = ncfO.membrane_constant(ch1, hyp_sweeps,hyp_onset)
inj, spike_counts, max_spikes, peaks, first_spike, IO_slope, ch1, dep_sweeps = ncfO.apcount(ch1, dep_sweeps, hyp_sweeps, dep_file, channel, deps)    
ax_spec = ncfO.APpanelplot(first_spike, inj)
Rheobase = ncfO.rheobase(inj, first_spike)
ivplot, subTHiv_fit, IR = ncfO.IV_curve(hyp_sweeps, first_spike, peak_deflection, ch1, Vm, onset)
TH, th_fig, d1 = ncfO.threshold(ch1, hyp_sweeps, inj, spike_counts, peaks, dep_sweeps, first_spike, max_spikes)
Firing_freq = ncfO.rel_firing_freq(first_spike, spike_counts)
max_firing_freq = ncfO.max_firing_freq(spike_counts)
AP1_max_deriv, AP1_min_deriv, slope_ratio = ncfO.slopes(TH, first_spike, d1, peaks)
mAHP = ncfO.mahp(dep_sweeps, ch1, hyp_sweeps, onset)
Latency = ncfO.latency(TH, inj, onset)
AP_height = ncfO.ap_height(peaks, first_spike, TH)
AP12, AP910, adaptn, ISI_res = ncfO.intervals(TH, spike_counts, max_spikes)
cut_AP = ncfO.cut_ap(TH, ch1, hyp_sweeps, spike_counts, first_spike,peaks,sweep=None, spike=1)
HW = ncfO.halfwidth(first_spike, TH, ch1, hyp_sweeps, spike_counts,peaks,sweep=None, spike=1)
AHP = ncfO.ahp(TH, ch1, hyp_sweeps, spike_counts, first_spike,peaks,sweep=None, spike=1)
#%%
# =============================================================================
# add single run cells to dictionary
# =============================================================================
notes = 'manually added'
Cell_ID=CellID   #use for when adding cells manually
count = len(cell_dic2.keys())
#tenAPs = np.where(spike_counts[:,1]>10)[0][0]
#Lat=Latency[tenAPs,1]
if max(spike_counts[:,1])<10:
    Lat = Latency[first_spike+2,1]
else:
    tenAPs = np.where(spike_counts[:,1]>10)[0][0]
    Lat=Latency[tenAPs,1]

new_entry = count+1
#enter values into dic
cell_dic2[new_entry] = {'CellID': Cell_ID, 'Vm':Vm, 
         'Sag Ratio':sag_ratio[-1], 'Membrane time constant': mc[3,0],
         'Membrane capacitance': mc[3,2], 'Input Resistance': IR[0], 
         'mAHP':mAHP[-1][2],'Latency': Latency[np.where(~np.isnan(Latency[:,1]))[0][0],1],
         'AP1-AP2int':AP12, 'AP9-AP10int': AP910, 'Adaptation': adaptn, 
         'AP height': AP_height, 'Max dV/dt': AP1_max_deriv, 'Min dV/dt': AP1_min_deriv, 
         'dV/dt ratio': slope_ratio, 'TH': TH[first_spike,0,3], 'AHP': AHP,
         'Halfwidth': HW, 'Rheobase': Rheobase, 'Rel. firing freq': Firing_freq, 
         'Max firing freq': max_firing_freq,'minISI': ISI_res, 'Rin':Ri, 
         'Rser':Ra, 'Holding current': holding_current,'Notes': notes, 
         'Membrane resistance': mc[1,1], 'Min latency': Lat}
gc.collect()

#%%
# =============================================================================
# Convert dictionary into DataFrame and transpose
# =============================================================================
cellDF = pd.DataFrame.from_dict(cell_dic2)
cellDF = cellDF.T
cellDF = cellDF.drop_duplicates(subset='CellID')
