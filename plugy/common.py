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
import sys
import random
import hashlib

__all__ = [
    'ROOT',
    '_const',
]

# get the location
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data')


try:
    basestring
except NameError:
    basestring = str

if 'long' not in __builtins__:
    long = int

if 'unicode' not in __builtins__:
    unicode = str


class _const:

    class ConstError(TypeError):

        pass

    def __setattr__(self, name, value):

        if name in self.__dict__:

            raise(self.ConstError, "Can't rebind const(%s)" % name)

        self.__dict__[name] = value


def gen_session_id(length = 5):
    """Generates a random alphanumeric string.

    :arg int length:
        Optional, ``5`` by default. Specifies the length of the random
        string.

    :return:
        (*str*) -- Random alphanumeric string of the specified length.
    """

    abc = '0123456789abcdefghijklmnopqrstuvwxyz'

    return ''.join(random.choice(abc) for i in xrange(length))


def md5(value):
    """
    Computes the sum of MD5 hash of a given string *value*.

    :arg str value:
        Or any other type (will be converted to string). Value for which
        the MD5 sum will be computed. Must follow ASCII encoding.

    :return:
        (*str*) -- Hash value resulting from the MD5 sum of the *value*
        string.
    """
    
    string = str(value).encode('ascii')
    
    return hashlib.md5(string).hexdigest()
