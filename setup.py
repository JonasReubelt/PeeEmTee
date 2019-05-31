#!usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: setup.py
"""
peeemtee setup script.

"""

from setuptools import setup, find_packages

PACKAGE_NAME = 'peeemtee'
URL = 'https://github.com/JonasReubelt/PeeEmTee'
DESCRIPTION = 'Auxiliary package for PMT analyses'
__author__ = 'Jonas Reubelt'
__email__ = 'jreubelt@km3net.de'

with open('requirements.txt') as fobj:
    REQUIREMENTS = [l.strip() for l in fobj.readlines()]

setup(
    name=PACKAGE_NAME,
    url=URL,
    description=DESCRIPTION,
    author=__author__,
    author_email=__email__,
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    setup_requires=['setuptools_scm'],
    use_scm_version={
        'write_to': '{}/version.txt'.format(PACKAGE_NAME),
        'tag_regex': r'^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$',
    },
    install_requires=REQUIREMENTS,
    python_requires='>=2.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
)
