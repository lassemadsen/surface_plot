#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

__version__ = "1.0.2"
NAME = 'surface_plot'
AUTHOR = "Center of Functionally Integrative Neuroscience"
MAINTAINER = "Lasse Stensvig Madsen"
EMAIL = 'lasse.madsen@clin.au.dk'
KEYWORDS = "brain MNI GPU visualization data OpenGL vispy neuroscience "
DESCRIPTION = "Visualization of cortical surfaces"
URL = ''
DOWNLOAD_URL = "https://github.com/lassemadsen/surface_plot/archive/" + \
               "v" + __version__ + ".tar.gz"
# Data path :
HERE = os.path.abspath(os.path.dirname(__file__))
PACKAGE_DATA = {'surface_plot': ['surface_data/*']}


def read(fname):
    """Read README"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    # DESCRIPTION
    name=NAME,
    version=__version__,
    description=DESCRIPTION,
    long_description=read('README.md'),
    keywords=KEYWORDS,
    license="BSD 3-Clause License",
    author=AUTHOR,
    maintainer=MAINTAINER,
    author_email=EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    # PACKAGE / DATA
    packages=find_packages(),
    package_dir={'surface_plot': 'surface_plot'},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    platforms='any',
    install_requires=[
        "visbrain @ https://github.com/lassemadsen/visbrain/archive/master.zip#egg=visbrain",
        "numpy>=1.13, <=1.26.4",
        "pandas",
        "brainspace>=0.1.2",
        "vtk==9.3.1",
        "scipy",
        "matplotlib>=3.3.0",
        "pillow"
    ],
    classifiers=["Development Status :: 3 - Alpha",
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 'Topic :: Scientific/Engineering :: Visualization',
                 "Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Operating System :: MacOS",
                 "Operating System :: POSIX :: Linux",
                 "Operating System :: Microsoft :: Windows",
                 "Natural Language :: English"
                 ])
