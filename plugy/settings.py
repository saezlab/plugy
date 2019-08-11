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


from future.utils import iteritems

import os
import collections

import plugy.common as common


_defaults = {
    # name of the module
    'module_name': 'plugy',
    # The absolute root directory.
    # This should not be necessary, why is it here?
    'path_root': '/',
    # The basedir for every files and directories in the followings.
    'basedir': os.getcwd(),
    # If None will be the same as ``basedir``.
    'progessbars': True,
    # verbosity for messages printed to console
    'console_verbosity': -1,
    # verbosity for messages written to log
    'log_verbosity': 0,
    # log flush time interval in seconds
    'log_flush_interval': 2,
    'data_basedir': None,
    'cachedir': None,
}

in_datadir = set()
in_cachedir = set()

def reset_all():
    
    settings = collections.namedtuple('Settings', list(_defaults.keys()))
    
    for k in _defaults.keys():
        
        val = getattr(defaults, k)
        
        if k in in_datadir:
            val = os.path.join(common.ROOT, 'data', val)
        
        setattr(settings, k, val)
    
    if settings.cachedir is None:
        
        settings.cachedir = os.path.join(
            os.path.expanduser('~'),
            '.plugy',
            'cache',
        )
    
    for k in in_cachedir:
        
        setattr(settings, k, os.path.join(settings.cachedir, _defaults[k]))
    
    globals()['settings'] = settings


def setup(**kwargs):
    
    for param, value in iteritems(kwargs):
        
        setattr(settings, param, value)


def get(param):
    
    if hasattr(settings, param):
        
        return getattr(settings, param)


def get_default(param):
    
    if hasattr(defaults, param):
        
        return getattr(defaults, param)


def reset(param):
    
    setup(param, get_default(param))


defaults = common._const()

for k, v in iteritems(_defaults):
    
    setattr(defaults, k, v)

reset_all()
