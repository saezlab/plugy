#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# This file is part of the `plugy` python module
#
# Copyright
# 2018-2021
# EMBL & Heidelberg University
#
# Author(s): Dénes Türei (turei.denes@gmail.com)
#            Nicolas Peschke
#            Olga Ivanova
#
# Distributed under the GPLv3 License.
# See accompanying file LICENSE.txt or copy at
#     http://www.gnu.org/licenses/gpl-3.0.html
#
# Webpage: https://github.com/saezlab/plugy
#


from setuptools import setup


with open('requirements.txt', 'r') as fp:

    requirements = [l.strip() for l in fp]

with open('README.md', 'r') as f:

    readme = f.read()


setup(
    name = 'plugy',
    version = '1.0.9',
    packages = ['plugy', 'plugy.data', 'plugy.test'],
    url = 'https://github.com/saezlab/plugy',
    license = 'GPLv3',
    author = 'Dénes Türei, Nicolas Peschke, Olga Ivanova',
    author_email = 'turei.denes@gmail.com',
    description = 'Processing plug microfluidics data',
    long_description = readme,
    long_description_content_type = 'text/markdown; charset=UTF-8',
    install_requires = requirements,
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)
