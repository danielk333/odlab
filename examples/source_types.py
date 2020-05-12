#!/usr/bin/env python

'''Example showing how SourceCollection's and Path's can be used to load data.

'''

import numpy as np
import pathlib

from pyod import SourceCollection, SourcePath


data_dir = '.' / pathlib.Path(__file__).parents[0] / 'example_data'


def example_wrapper(text):
    def genereated_wrapper(func):
        def display_example(*args, **kwargs):
            eg_str = '=== Example: {} ==='.format(text)
            print('='*len(eg_str))
            print(eg_str)
            print('='*len(eg_str))

            ret = func(*args, **kwargs)

            print('='*len(eg_str) + '\n')

            return ret
        return display_example
    return genereated_wrapper

@example_wrapper('Loading sources from RAM')
def example_ram_data():
    data = [{
            'data': np.array([]),
            'meta': {},
            'index': 42,
        },
        {
            'data': np.array([]),
            'meta': {},
            'index': 43,
        }
    ]

    paths = SourcePath.from_list(data, 'ram')

    for path in paths:
        print(path)

    sources = SourceCollection(paths = paths)
    sources.details()



@example_wrapper('Parsing folders recursively for data files')
def example_recursive_file_find():

    paths = SourcePath.recursive_folder(data_dir, ['tdm', 'oem'])

    for path in paths:
        print(path)


@example_wrapper('Using glob to find data files')
def example_glob_file_find():

    paths = SourcePath.from_glob(data_dir / '*.oem')

    for path in paths:
        print(path)


@example_wrapper('Using glob to create a source collection')
def example_glob_load():
    sources = SourceCollection(paths = SourcePath.from_glob(data_dir / '*'))
    sources.details()


if __name__ == '__main__':
    example_ram_data()
    example_recursive_file_find()
    example_glob_file_find()
    example_glob_load()