#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  This file is part of the `plugy` python module
#
#  Copyright
#  2018-2019
#  EMBL, Heidelberg University
#
#  File author(s): Dénes Türei (turei.denes@gmail.com)
#                  Nicolas Peschke
#
#  Distributed under the GPLv3 License.
#  See accompanying file LICENSE.txt or copy at
#      http://www.gnu.org/licenses/gpl-3.0.html
#


import os
import collections

import plugy.session as session


SampleBase = collections.namedtuple(
    'SampleBase',
    [
        'drug1',
        'drug2',
        'count',
        'cells',
        'assay',
        'valve1',
        'valve2',
        'original_label',
    ],
)


class Sample(SampleBase):
    
    @property
    def label(self):
        
        drugs = sorted([self.drug1, self.drug2])
        
        return (
            'Barcode'
                if self.is_barcode else
            'Control'
                if self.is_control else
            ' & '.join(drug for drug in drugs if drug != 'Empty')
        )
    
    
    @property
    def is_barcode(self):
        
        return (
            self.drug1.startswith('Barcode') or
            self.drug2.startswith('Barcode')
        )
    
    @property
    def is_control(self):
        
        return self.drug1 == 'Empty' and self.drug2 == 'Empty'


class Sequence(session.Logger):
    
    
    def __init__(
            self,
            seq_file,
            drug_file,
        ):
        
        session.Logger.__init__(self, name = 'sequence')
        
        self.seq_file = seq_file
        self.drug_file = drug_file
        
        self.main()
    
    
    def main(self):
        
        self.read_drugs()
        self.read_sequence()
    
    
    def read_drugs(self):
        
        self.drugs = {}
        
        self._log('Reading drug-valve pairs from `%s`.' % self.drug_file)
        
        with open(self.drug_file, 'r') as fp:
            
            for line in fp:
                
                line = line.strip().split(';')
                
                self.drugs[int(line[0])] = line[2]
    
    
    def read_sequence(self):
        
        self.sequence = []
        
        if not os.path.exists(self.seq_file):
            
            self._log('File not found: `%s`.' % self.seq_file)
            return
        
        self._log('Reading sample sequence from `%s`.' % self.seq_file)
        
        with open(self.seq_file, 'r') as fp:
            
            for line in fp:
                
                line = line.strip().split(',')
                
                valve1 = int(line[5])
                valve2 = int(line[6])
                drug1 = self.valve_to_drug(valve1)
                drug2 = self.valve_to_drug(valve2)
                
                self.sequence.append(
                    Sample(
                        drug1 = self.drugs[valve1],
                        drug2 = self.drugs[valve2],
                        count = int(line[1]),
                        assay = int(line[4]) == 10,
                        cells = int(line[3]) == 9,
                        valve1 = valve1,
                        valve2 = valve2,
                        original_label = line[2],
                    )
                )
    
    
    def valve_to_drug(self, valve):
        
        if valve not in self.drugs:
            
            self._log('Valve %u missing from drugs list!' % valve)
        
        return self.drugs[valve] if valve in self.drugs else 'Unknown'
    
    
    def __iter__(self):
        
        for sample in self.sequence:
            
            yield sample
