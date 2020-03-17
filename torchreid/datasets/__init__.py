# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cuhk03 import CUHK03


__imgreid_factory = {
    'cuhk03': CUHK03,
}

def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)
