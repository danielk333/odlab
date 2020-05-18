import os
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess

class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)

with open('README', 'r') as fh:
    long_description = fh.read()

with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req if len(x.strip()) > 0]

setuptools.setup(
    name='pyod',
    version='0.1.0',
    long_description=long_description,
    url='https://github.com/danielk333/pyod',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MPL 2.0',
        'Operating System :: OS Independent',
    ],
    install_requires=pip_req,
    packages=setuptools.find_packages(),
    package_data={},
    # metadata to display on PyPI
    author='Daniel Kastinen',
    author_email='daniel.kastinen@irf.se',
    description='Python Orbit Determination',
    license='Mozilla Public License Version 2.0',
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
