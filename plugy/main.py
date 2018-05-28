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

import skimage.filters
from scipy import ndimage as ndi


class Plugy(object):
    
    colors = {
        'green':  '#5D9731',
        'blue':   '#3A73BA',
        'orange': '#F68026'
    }
    
    def __init__(self,
                 infile,
                 cut = (None, None),
                 drugs = [],
                 signal_threshold = .02,
                 adaptive_signal_threshold = True,
                 peak_minwidth = 5,
                 channels = {
                     'barcode': ('blue', 3),
                     'cells':   ('orange', 2),
                     'readout': ('green', 1)
                 },
                 bc_mean_peaks = 1,
                 discard = (2, 1),
                 gaussian_smoothing_sigma = 33,
                 adaptive_threshold_blocksize = 111,
                 adaptive_threshold_method = 'gaussian'
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
            :param bool adaptive_signal_threshold:
                Apply adaptive thresholding to identify peaks. This may help if
                the plugs are so close to each other that the signal from one or
                more channels do not drop below the `signal_threshold` value at
                their boundaries.
            :param int peak_minwidth:
                Minimum width of a peak in data points.
                E.g. if acquisition rate is 300 Hz a width of 300 means 1 second.
            :param dict channels:
                A dict of channels with channel names as keys and tuples of
                color and column index as values.
            :param int gaussian_smoothing_sigma:
                Sigma parameter for the gaussian curve used for smoothing before
                adaptive thresholding.
            :param int adaptive_threshold_blocksize:
                Blocksize for adaptive thresholding.
                Passed to `skimage.filters.threshold_local`.
            :param str adaptive_threshold_method:
                Method for adaptive thresholding.
                Passed to `skimage.filters.threshold_local`.
        
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
        self.set_channels(channels)
        
        for k, v in locals().items():
            
            if not hasattr(self, k):
                
                setattr(self, k, v)
        #self.cut = cut
        #self.drugs = drugs
        #self.peak_minwidth = peak_minwidth
        #self.signal_threshold = signal_threshold
        #self.adaptive_signal_threshold = adaptive_signal_threshold
        #self.bc_min_peaks = bc_mean_peaks
        #self.discard = discard
        #self.gaussian_smoothing_sigma = gaussian_smoothing_sigma
        #self.adaptive_threshold_blocksize = adaptive_threshold_blocksize
        #self.adaptive_threshold_method = adaptive_threshold_method
    
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
        
        self.peaks()
        self.samples()
    
    def peaks(self):
        """
        First part of the workflow. Identifies peaks and plots them.
        Running this first is useful to investigate the plot and find
        out optimal values for the `cut` parameter and the method to
        detect cycles. Calling `main()` runs both this part and the
        second part (see `samples()`).
        """
        self.read()
        self.strip()
        self.find_peaks()
        self.peaks_df()
        self.plot_peaks()
        self.plot_peaks(raw = True)
    
    def samples(self):
        """
        Second part of the workflow. Identifies samples, builds a data
        frame and exports the preprocessed data.
        """
        
        self.sample_names()
        self.samples_df()
        self.export()
    
    def read(self):
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
    
    def set_channels(self, channels):
        
        self.channels  = channels
        self.channelsl = [
            x[0] for x in
            sorted(
                self.channels.values(),
                key = lambda x: x[1]
            )
        ]
    
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
        
        # peak segmentation first by fix threshold
        self.peaksa = np.any(self.data[:,1:4] > self.signal_threshold, 1)
        
        # then optionally by adaptive threshold
        if self.adaptive_signal_threshold:
            
            at_channels = []
            
            for color, i in self.channels.values():
                
                this_channel = self.data[:,i]
                # smooth before thresholding
                this_channel = ndi.filters.gaussian_filter(
                    this_channel,
                    sigma = self.gaussian_smoothing_sigma
                )
                # skimage.filters.threshold_local works only on 2D arrays
                this_channel.shape = (this_channel.shape[0], 1)
                
                this_at = skimage.filters.threshold_local(
                    this_channel,
                    self.adaptive_threshold_blocksize,
                    method = self.adaptive_threshold_method #,
                    #param  = 50 # slightly larger sigma than default
                )
                # adaptive and fix thresholds are combined
                at_channels.append(
                    np.logical_and(
                        this_channel > this_at,
                        this_channel > self.signal_threshold
                    )
                )
                
                # self.data[:,i] = this_channel[:,0]
            
            # then we combine the detected peaks from all channels
            self.peaksa = np.any(np.hstack(at_channels), 1)
        
        # indices of peak starts and ends
        startend = np.where(self.peaksa[1:] ^ self.peaksa[:-1])[0] + 1
        
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
            columns = ['i0', 'i1', 't0', 't1'] + self.channelsl
        )
        self.peakdf_n = pd.melt(
            self.peakdf,
            ['i0', 'i1', 't0', 't1'],
            self.channelsl,
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
    
    def plot_peaks(self, pdf_png = 'pdf', raw = False):
        """
        Creates a plot with median intensities of each channel
        from each peak.
        
        :param str pdf_png: File type. Default is `pdf`, alternative is `png`.
        :param bool raw: Plot not only peak medians but also raw data.
        """
        
        pdf_png = 'png' if pdf_png == 'png' or raw else 'pdf'
        
        fname = '%s.raw.%s' % (self.infile, pdf_png)
        
        sys.stdout.write(
            '\t:: Plotting median intensities into\n\t   `%s`.\n' % fname
        )
        
        fig = mpl.figure.Figure(figsize = (300, 3))
        
        if pdf_png == 'pdf':
            
            pdf = mpl.backends.backend_pdf.PdfPages(fname)
            cvs = mpl.backends.backend_pdf.FigureCanvasPdf(fig)
            
        else:
            # else means png
            cvs = mpl.backends.backend_agg.FigureCanvasAgg(fig)
        
        ax = fig.add_subplot(111)
        
        if raw:
            
            for color, i in self.channels.values():
                
                ax.plot(
                    self.data[:,0],
                    self.data[:,i],
                    c = self.colors[color],
                    zorder = 1
                )
            
            ymax = np.max(
                self.data[:,
                    min([x[1] for x in self.channels.values()]):
                    max([x[1] for x in self.channels.values()]) + 1
                ]
            )
            
            # highlighting plugs
            bc_sample = (
                ndi.filters.gaussian_filter(
                    self.data[:,self.channels['barcode'][1]],
                    13
                ) >
                np.vstack((
                    ndi.filters.gaussian_filter(
                        self.data[:,self.channels['cells'][1]],
                        13
                    ),
                    ndi.filters.gaussian_filter(
                        self.data[:,self.channels['readout'][1]],
                        13
                    )
                )).max(0)
            )
            
            ax.fill_between(
                self.data[:,0],
                0,
                ymax,
                where = np.logical_and(
                    self.peaksa, bc_sample
                ),
                facecolor = '#C8CADF', # light blue
                zorder = 0
            )
            
            ax.fill_between(
                self.data[:,0],
                0,
                ymax,
                where = np.logical_and(
                    self.peaksa, np.logical_not(bc_sample)
                ),
                facecolor = '#FBD1A7', # light orange
                zorder = 0
            )
        
        # scatterplots for the peak medians of each channel
        for color, i in self.channels.values():
            
            ax.scatter(
                self.peakdf.t0,
                getattr(self.peakdf, color),
                c = self.colors[color],
                edgecolors = '#000000' if raw else 'none',
                linewidths = 1,
                label = color.upper(),
                s = 12,
                zorder = 2
            )
        
        _ = ax.set_xlabel('Time (s)')
        _ = ax.set_ylabel('Fluorescence\nintensity')
        
        fig.tight_layout()
        cvs.draw()
        
        if pdf_png == 'pdf':
            
            cvs.print_figure(pdf)
            pdf.close()
            
        else:
            
            cvs.print_png(fname)
        
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
