#!/usr/bin/env python

'''Test CCSDS data format load functions

'''

import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import pathlib

import unittest
import numpy as np
import numpy.testing as nt

import pyod.sources as src


data_dir = str('.' / pathlib.Path(__file__).parents[0] / 'data')


class TestSources(unittest.TestCase):
    
    def test_ABC_exception(self):
        with self.assertRaises(NotImplementedError):
            obj = src.ObservationSource(path = None)

        with self.assertRaises(NotImplementedError):
            obj = src.TrackletSource(path = None)

        with self.assertRaises(NotImplementedError):
            obj = src.StateSource(path = None)

class TestSourcePath(unittest.TestCase):
    
    def test_init(self):
        for ptype in src._ptypes:
            obj = src.SourcePath(data = None, ptype=ptype)

        with self.assertRaises(TypeError):
            obj = src.SourcePath(data = None, ptype='This type does not exist')

    def test_str(self):
        for ptype in src._ptypes:
            if ptype == 'ram':
                obj = src.SourcePath(data = {'data': None}, ptype=ptype)
            else:
                obj = src.SourcePath(data = None, ptype=ptype)
            str_ = str(obj)


    def test_recursive_folder(self):
        paths = src.SourcePath.recursive_folder(data_dir + '/', exts=['oem', 'tdm'])
        assert len(paths) == 2
        for path in paths:
            assert path.ptype == 'file'

    def test_from_glob(self):
        paths = src.SourcePath.from_glob(data_dir + '/*.oem')
        assert len(paths) == 1
        paths = src.SourcePath.from_glob(data_dir + '/*')
        assert len(paths) == 3
        for path in paths:
            assert path.ptype == 'file'
        paths = src.SourcePath.from_glob(data_dir + '/tracklet.[h]*')
        assert len(paths) == 1


    def from_list(self):
        paths = src.SourcePath.from_list(data=range(10), ptype='ram')
        assert len(paths) == 10
        for path in paths:
            assert path.ptype == 'ram'


class TestOrbitEphemerisMessageSource(unittest.TestCase):

    def setUp(self):
        self.path = src.SourcePath(data = data_dir + '/state.oem', ptype='file')

    def test_init(self):
        obj = src.OrbitEphemerisMessageSource(path = self.path)

    def test_accept(self):
        assert src.OrbitEphemerisMessageSource.accept(self.path)
        with self.assertRaises(TypeError):
            src.OrbitEphemerisMessageSource.accept(None)
        assert not src.OrbitEphemerisMessageSource.accept(src.SourcePath(data = data_dir + '/tracklet.tdm', ptype='file'))
        assert not src.OrbitEphemerisMessageSource.accept(src.SourcePath(data = {'data': None}, ptype='ram'))

    def test_load(self):
        obj = src.OrbitEphemerisMessageSource(path = self.path)

        assert obj.index == 4750

        assert len(obj.meta) == 15, print(obj.meta)

        assert 'REF_FRAME' in obj.meta

        for met_ in src.OrbitEphemerisMessageSource.REQUIRED_META:
            assert met_ in obj.meta

        assert len(obj.data) == 4

        for dt in src.OrbitEphemerisMessageSource.dtype:
            assert dt[0] in obj.data.dtype.names
    
        nt.assert_almost_equal(obj.data[0]['x'], -7100297.113, decimal=3)



class TestTrackingDataMessageSource(unittest.TestCase):

    def setUp(self):
        self.path = src.SourcePath(data = data_dir + '/tracklet.tdm', ptype='file')

    def test_init(self):
        obj = src.TrackingDataMessageSource(path = self.path)

    def test_accept(self):
        assert src.TrackingDataMessageSource.accept(self.path)
        with self.assertRaises(TypeError):
            src.TrackingDataMessageSource.accept(None)
        assert not src.TrackingDataMessageSource.accept(src.SourcePath(data = data_dir + '/state.oem', ptype='file'))
        assert not src.TrackingDataMessageSource.accept(src.SourcePath(data = {'data': None}, ptype='ram'))

    def test_load(self):
        obj = src.TrackingDataMessageSource(path = self.path)

        assert obj.index == 1001120300074

        assert len(obj.meta) == 25, print(obj.meta)

        assert 'PATH' in obj.meta

        for met_ in src.TrackingDataMessageSource.REQUIRED_META:
            assert met_ in obj.meta

        assert len(obj.data) == 38

        for dt in src.TrackingDataMessageSource.dtype:
            assert dt[0] in obj.data.dtype.names
    
        nt.assert_almost_equal(obj.data[0]['r'], 10539.162206964465e3, decimal=5)