#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Denes Turei EMBL 2018
# turei.denes@gmail.com

from __future__ import print_function
from future.utils import iteritems
from past.builtins import xrange, range, reduce

import sys
import imp
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.figure
import matplotlib.backends.backend_pdf


class Plugy(object):
    
    def __init__(self,
                 infile,
                 cut = (None, None),
                 drugs = [],
                 signal_threshold = 0.02,
                 peak_minwidth = 5,
                 channels = {
                     'barcode': ('blue', 2),
                     'cells':   ('orange', 3),
                     'readout': ('green', 1)
                 },
                 bc_mean_peaks = 1,
                 discard = (2, 1)
        ):
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
        
        Example:
        
        >>> import plugy
        
        >>> drugs = [
            '11:Void', '12:Void', '13:Nutlin', '14:Cyt-387', '15:IKK16',
            '16:MK-2206', '17:IL-6', '18:Gefitinib', '19:IFN-γ',
            '20:Soratinib', '21:TGF-β', '22:Dasatinib'
        ]
        
        >>> p = plugy.Plugy(
            infile = 'example_screen.txt',
            cut = (3225, 11200),
            drugs = drugs
        )
        
        >>> p.main()
        """
        
        self.infile = infile
        self.channels = ['green', 'orange', 'blue']
        self.cut = cut
        self.drugs = drugs
        self.peak_minwidth = peak_minwidth
        self.signal_threshold = signal_threshold
        self.bc_min_peaks = bc_mean_peaks
        self.discard = discard
    
    def reload(self):
        """
        Reloads the module and updates the instance.
        Use this to make updates on this code while
        preserving the actual instance and its state.
        """
        
        modname = self.__class__.__module__
        mod = __import__(modname, fromlist=[modname.split('.')[0]])
        imp.reload(mod)
        new = getattr(mod, self.__class__.__name__)
        setattr(self, '__class__', new)
    
    def main(self):
        """
        Executes all steps of the workflow.
        Reads the data, removes unnecessary parts from the beginning
        and the end, identifies peaks (plugs), measures their median
        intensities in each channel, identifies samples and barcodes,
        assigns drug combinations to samples, plots the medians
        and exports the preprocessed data.
        """
        
        self.reader()
        self.strip()
        self.find_peaks()
        self.peaks_df()
        self.plot_peaks()
        self.sample_names()
        self.samples_df()
        self.export()
    
    def reader(self):
        """
        Reads data from `infile`. It starts reading after the line
        starting with `Time` or `X_Value`. But check your file to
        be sure it has this line at the beginning of data and nowhere
        else above. These files have unexpected variations in their
        format.
        """
        
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
        """
        Removes beginning and end sections from data as it has been
        provided in the `cut` parameter. For example, if
        `cut = (3000, 33000)`, data before 3000 s and after 30000 s
        will be removed. This way you can remove empty sections and
        data from priming plugs. If nothing is to be removed provide
        `None`, e.g. `cut = (3000, None)` won't remove anything from
        the end.
        """
        
        if self.cut[0] is not None:
            
            self.data = self.data[np.where(self.data[:,0] > self.cut[0])[0],:]
        
        if self.cut[1] is not None:
            
            self.data = self.data[np.where(self.data[:,0] < self.cut[1])[0],:]
    
    def find_peaks(self):
        """
        Finds peaks i.e. plugs.
            * Whereever any channel is above the `signal_threshold` is
              considered to be a peak segment
            * Then indices corresponding to start and end of each peak
              segment are determined. These are stored in the `startend`
              array where first column contains the start indices and
              second column the end indices.
            * Finally data from all peaks are collected into the `peakval`
              array. In this array first two columns are the start and
              end indices, columns 3-4 are the start and end times in
              seconds, while following columns are the medians of
              each channel values within the peak.
        """
        
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
        """
        Creates a `pandas.DataFrame` from the `peakval` array.
        This is to be stored in the `peakdf` attribute.
        Then it creates a long format data frame, find it
        under the `peakdf_n` attribute.
        """
        
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
        """
        Iterates through the `peakdf` data frame and processes
        samples. Samples are a series of peaks (plugs) with
        certain treatment conditions (drug combination).
        Extends the `peakdf` data frame with the following
        columns:
            * `barcode`: The peak is a barcode peak.
            * `sample`: The peak is a sample peak (should
              be simply negation of `barcode`)
            * `barcode_id`: The sequence number of the barcode
              segment (segment is a series of barcode plugs);
              0 means this plug is not a barcode
            * `sample_id`: Same for samples; plugs with the
              same number in this column belong to the same
              condition, 0 means the plug is not a sample but
              barcode
            * `drug1`: Name of first drug
            * `drug2`: Name of second drug
            * `drugs`: Names of the drugs separated by `_`
            * `runs`: Replicate number, i.e. if you repeated
              the the sequence 3 times, it will be 0, 1, 2
            * `discard`: Whether the plug should be discarded.
              Usually we discard first and last plugs in each
              sample, by default `discard` attribute is `(2, 1)`
              which means first 2 and the last plug are to be
              discarded.
        """
        
        insample  = self.peakdf.orange > self.peakdf.blue
        inbarcode = np.logical_not(insample)
        
        bci = 0 # number of current barcode segment
        smi = 0 # number of current sample segment
        run = 0 # number of current run
        bca = [] # vector with barcode segment ids; 0 means not barcode
        sma = [] # vector with sample segment ids; 0 means not sample
        bcb = False # boolean value tells if current segment is barcode
        smb = False # boolean value tells if current segment is sample
        dr1 = [] # vector with names of drug #1
        dr2 = [] # vector with names of drug #2
        drs = [] # vector with names of drug combinations
        rns = [] # vector with run numbers
        dsc = [] # vector with boolean values wether the peak should 
                 # be discarded
        
        for i in xrange(self.peakdf.shape[0]):
            
            if inbarcode[i]:
                
                if not bcb:
                    
                    bcb = True
                    smb = False
                    bci += 1
                    bcl = 0
                    
                    if len(dsc) >= self.discard[1]:
                        # discard last sample peaks
                        dsc[-self.discard[1]:] = [True] * self.discard[1]
                
                bca.append(bci)
                bcl += 1
                dsc.append(False)
                rns.append(run)
                
            else:
                
                bca.append(0)
            
            if insample[i]:
                
                if not smb:
                    
                    if bcl <= self.bc_min_peaks and bcb:
                        
                        bci -= 1
                        bca[-bcl:] = [0] * bcl
                    
                    else:
                        
                        smi += 1
                        run = smi // (len(self.samples_drugs))
                        sml = 0
                    
                    smb = True
                    bcb = False
                
                sma.append(smi)
                # discard first sample peaks
                dsc.append(sml < self.discard[0])
                sml += 1
                rns.append(run)
                dr1.append(self.samples_drugs[
                    smi % (len(self.samples_drugs))][0])
                dr2.append(self.samples_drugs[
                    smi % (len(self.samples_drugs))][1])
                drs.append('%s_%s' % (
                    self.samples_drugs[smi % (len(self.samples_drugs))][0],
                    self.samples_drugs[smi % (len(self.samples_drugs))][1]
                ))
                
            else:
                
                sma.append(0)
                dr1.append('NA')
                dr2.append('NA')
                drs.append('NA')
        
        self.peakdf['barcode']    = pd.Series(inbarcode)
        self.peakdf['sample']     = pd.Series(insample)
        self.peakdf['barcode_id'] = pd.Series(np.array(bca))
        self.peakdf['sample_id']  = pd.Series(np.array(sma))
        self.peakdf['drug1']      = pd.Series(np.array(dr1))
        self.peakdf['drug2']      = pd.Series(np.array(dr2))
        self.peakdf['drugs']      = pd.Series(np.array(drs))
        self.peakdf['runs']       = pd.Series(np.array(rns))
        self.peakdf['discard']    = pd.Series(np.array(dsc))
    
    def plot_peaks(self):
        """
        Creates a plot with median intensities of each channel
        from each peak.
        """
        
        pdfname = '%s.raw.pdf' % self.infile
        
        sys.stdout.write(
            '\t:: Plotting median intensities into\n\t   `%s`.\n' % pdfname
        )
        
        pdf = mpl.backends.backend_pdf.PdfPages(pdfname)
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
        """
        From a linear list of drug names creates the sequence of
        drug combinations according to the way it was in Fede's
        paper and in the BraDiPluS R package.
        The list of drugs you provided in the `drugs` parameter.
        This list should correspond to the sequence how the drugs
        have been connceted to the valves.
        Then iterate the list of drugs and for each drug we iterate
        the drugs before this drug in the list to get non redundant
        drug pairs.
        
        Important: Please override this method if you used different
        drug sequence!
        """
        
        sys.stdout.write(
            '\t:: Assuming sample (valve pair opening) sequence '
            'used by Fede.\n\t   Please override the `sample_names` '
            'method if you used different sequence.\n'
        )
        
        names = []
        
        for i in xrange(1, len(self.drugs)):
            
            for j in xrange(1, i):
                
                if not len(names) or (len(names) + 1) % 11 == 0:
                    
                    names.append((self.drugs[0], self.drugs[1]))
                
                names.append((self.drugs[j], self.drugs[i]))
        
        names.append((self.drugs[0], self.drugs[1]))
        
        self.samples_drugs = names
    
    def export(self, outfile = None):
        """
        Writes the `peaksdf` data frame into tsv file.
        """
        
        outfile = outfile or '%s.peaks.tsv' % self.infile
        
        sys.stdout.write(
            '\t:: Exporting peaks data into\n\t   `%s`\n' % outfile
        )
        
        self.peakdf.to_csv(outfile, sep = '\t', index = False)
