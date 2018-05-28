#!/usr/bin/env Rscript

# Dénes Türei EMBL 2018
# turei.denes@gmail.com

import plugy

drugs = [
    'MK & PHT',  # 13, 14
    'PHT & Inh', # 13, 15
    'PHT & St',  # 13, 16
    'MK & Inh',  # 14, 15
    'MK & St', # 14, 16
    'St & Inh', # 15, 16
    'PHT',
    'MK',
    'Inh',
    'St'
]

p = plugy.Plugy(
    infile = 'denes_martine_1.txt',
    cut = (580, 4600),
    drugs = drugs,
    direct_drug_combinations = True
)

p.main()
