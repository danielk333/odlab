#!/usr/bin/env python

'''Example showing how SourceCollection's and Path's can be used to load data.

'''
import pathlib
import odlab

print("Supported sources:")
print(odlab.SOURCES.keys())

data_dir = (pathlib.Path(__file__).parent / 'example_data').resolve()
tdm_files = list(data_dir.glob("*.tdm"))
hdf5_files = list(data_dir.glob("*.h5"))

print("TDM radar data")
tdm_data = odlab.load_source(tdm_files[0], "radar_text_tdm")
print(tdm_data)

print("HDF5 radar data")
h5_data = odlab.load_source(hdf5_files[0], "radar_hdf")
print(h5_data)


# Or trough the convince mass loader

dfs = odlab.glob_sources(
    data_dir, 
    {
        "radar_text_tdm": "*.tdm",
        "radar_hdf": "*.h5",
    }
)

print(f"{len(dfs)} files loaded")
print(f"File 2 ({dfs[1].attrs['path']}):")
print(dfs[1])
