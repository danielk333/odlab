#!/usr/bin/env python

'''Example showing how SourceCollection's and Path's can be used to load data.

'''

from pyod import SourceCollection, Path

eg_str = '=== Testing "{}" source ==='

print(eg_str.format('RAM'))

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

paths = Path.from_list(data, 'ram')

for path in paths:
    print(path)

sources = SourceCollection(paths = paths)
sources.details()


print('='*len(eg_str.format('RAM')))
print(eg_str.format('tdm + oem folder search'))


paths = Path.recursive_folder('./tests/tmp_test_data/test_sim/master', ['tdm', 'oem'])

print('RECURSIVE')
for path in paths:
    print(path)


print('='*len(eg_str.format('tdm + oem folder search')))
print(eg_str.format('glob file search'))


path_pri = './tests/tmp_test_data/test_sim/master/prior'
paths = Path.from_glob(path_pri)

print('SINGLE GLOB')
for path in paths:
    print(path)

sources = SourceCollection(paths = paths)
sources.details()

