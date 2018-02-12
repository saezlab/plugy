#!/usr/bin/env python

# Denes Turei EMBL 2018
# turei.denes@gmail.com

from __future__ import print_function
from future.utils import iteritems
from past.builtins import xrange, range, reduce

import imp
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.figure
import matplotlib.backends.backend_pdf


class BDQuick(object):
    
    def __init__(self, infile = '20180124_pancreas-tumor_denes3.txt',
                 cut = (None, None), drugs = [],
                 signal_threshold = 0.02,
                 peak_minwidth = 5,
                 channels = {
                     'barcode': ('blue', 2),
                     'cells':   ('orange', 3),
                     'readout': ('green', 1)
                 },
                 bc_mean_peaks = 1):
        """
        This object represents a plug based microfluidics screen.
        
        Args:
            :param str infile:
                Name of the file containing the acquired data.
            :param tuple cut:
                A segment of the data along the time axis to be
                used. E.g. `(800, 9000)` the data points before 800 s and after
                9000 s will be removed.
            :param list drugs:
                List of the compounds connected to inlets 11-22
                of the BD chip.
            :param float signal_threshold:
                Threshold to be used at the selection
                of peaks. Values in any channel above this will be considered as
                a peak (plug).
            :param int peak_minwidth:
                Minimum width of a peak in data points.
                E.g. if acquisition rate is 300 Hz a width of 300 means 1 second.
            :param dict channels:
                A dict of channels with channel names as keys and tuples of
                color and column index as values.
        """
        
        self.infile = infile
        self.channels = ['green', 'orange', 'blue']
        self.cut = cut
        self.drugs = drugs
        self.peak_minwidth = peak_minwidth
        self.signal_threshold = signal_threshold
        self.bc_min_peaks = bc_mean_peaks
    
    def reload(self):
        
        modname = self.__class__.__module__
        mod = __import__(modname, fromlist=[modname.split('.')[0]])
        imp.reload(mod)
        new = getattr(mod, self.__class__.__name__)
        setattr(self, '__class__', new)
    
    def main(self):
        
        self.reader()
        self.strip()
        self.find_peaks()
        self.peaks_df()
        self.plot_peaks()
        self.sample_names()
        self.samples_df()
    
    def reader(self):
        
        self.data = []
        
        with open(self.infile, 'r') as fp:
            
            for l in fp:
                
                if l[1:].strip().startswith('Time') or l.startswith('X_Value'):
                    
                    break
            
            for l in fp:
                
                l = l.replace(',', '.').strip().split('\t')
                self.data.append([float(i) for i in l])
        
        self.data = np.array(self.data)
    
    def strip(self):
        
        if self.cut[0] is not None:
            
            self.data = self.data[np.where(self.data[:,0] > self.cut[0])[0],:]
        
        if self.cut[1] is not None:
            
            self.data = self.data[np.where(self.data[:,0] < self.cut[1])[0],:]
    
    def find_peaks(self):
        
        # peak segments
        self.peaks = np.any(self.data[:,1:4] > self.signal_threshold, 1)
        
        # indices of peak starts and ends
        startend = np.where(self.peaks[1:] ^ self.peaks[:-1])[0] + 1
        
        if len(startend) % 2 == 1:
            
            startend = startend[:-1]
        
        startend = np.vstack([
            startend[range(0, len(startend), 2)],
            startend[range(1, len(startend), 2)]
        ]).T
        
        startend = startend[
            np.where(startend[:,1] - startend[:,0] >= self.peak_minwidth)
        ]
        
        peakval = []
        
        for se in startend:
            
            peakval.append([
                se[0], # start index
                se[1], # end index
                np.min(self.data[se[0]:se[1],0]), # the start time
                np.max(self.data[se[0]:se[1],0])  # the end time
                ] + [
                np.median(self.data[se[0]:se[1],i])
                for i in range(1, 4)
            ])
            
        self.peakval = np.array(peakval)
        self.startend = startend
    
    def peaks_df(self):
        
        self.peakdf = pd.DataFrame(
            self.peakval,
            columns = ['i0', 'i1', 't0', 't1'] + self.channels
        )
        self.peakdf_n = pd.melt(
            self.peakdf,
            ['i0', 'i1', 't0', 't1'],
            self.channels,
            'channel',
            'value'
        )
    
    def samples_df(self):
        
        insample  = self.peakdf.orange > self.peakdf.blue
        inbarcode = np.logical_not(insample)
        
        bci = 0
        smi = 0
        bca = []
        sma = []
        bcb = False
        smb = False
        dr1 = []
        dr2 = []
        
        for i in xrange(self.peakdf.shape[0]):
            
            if inbarcode[i]:
                
                if not bcb:
                    
                    bcb = True
                    smb = False
                    bci = bci + 1
                    bcl = 0
                
                bca.append(bci)
                bcl += 1
                
            else:
                
                bca.append(0)
            
            if insample[i]:
                
                if not smb:
                    
                    if bcl <= self.bc_min_peaks and bcb:
                        
                        bci -= 1
                        bca[-bcl:] = [0] * bcl
                    
                    else:
                        
                        smi += 1
                    
                    smb = True
                    bcb = False
                
                sma.append(smi)
                dr1.append(self.samples_drugs[
                    smi % (len(self.samples_drugs) + 1)][0])
                dr2.append(self.samples_drugs[
                    smi % (len(self.samples_drugs) + 1)][1])
                
            else:
                
                sma.append(0)
                dr1.append('NA')
                dr1.append('NA')
        
        self.peakdf['barcode']    = pd.Series(inbarcode)
        self.peakdf['sample']     = pd.Series(insample)
        self.peakdf['barcode_id'] = pd.Series(np.array(bca))
        self.peakdf['sample_id']  = pd.Series(np.array(sma))
    
    def plot_peaks(self):
        
        pdf = mpl.backends.backend_pdf.PdfPages('%s.raw.pdf' % self.infile)
        fig = mpl.figure.Figure(figsize = (300, 3))
        cvs = mpl.backends.backend_pdf.FigureCanvasPdf(fig)
        
        ax = fig.add_subplot(111)
        ax.scatter(self.peakdf.t0, self.peakdf.green,
                   c = '#5D9731', label = 'Green', s = 2)
        ax.scatter(self.peakdf.t0, self.peakdf.blue,
                   c = '#3A73BA', label = 'Blue', s = 2)
        ax.scatter(self.peakdf.t0, self.peakdf.orange,
                   c = '#F68026', label = 'Orange', s = 2)
        _ = ax.set_xlabel('Time (s)')
        _ = ax.set_ylabel('Fluorescence\nintensity')
        
        fig.tight_layout()
        cvs.draw()
        cvs.print_figure(pdf)
        pdf.close()
        fig.clf()
    
    def sample_names(self):
        
        names = []
        
        for i in xrange(1, len(self.drugs)):
            
            for j in xrange(1, i):
                
                if not len(names) or (len(names) + 1) % 11 == 0:
                    
                    names.append((self.drugs[0], self.drugs[1]))
                
                names.append((self.drugs[j], self.drugs[i]))
        
        names.append((self.drugs[0], self.drugs[1]))
        
        self.samples_drugs = names
    
    def export(self, outfile = ''):
        
        pass
