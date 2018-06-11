from setuptools import setup

import peeemtee

__author__ = 'Jonas Reubelt'
VERSION = "0.0.1"

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

setup(name='peeemtee',
      version=VERSION,
      url='https://github.com/JonasReubelt/PeeEmTee',
      description='PeeEmTee',
      author=__author__,
      author_email='jonas.reujon.reubelt@fau.de',
      packages=['peeemtee'],
      include_package_data=True,
      platforms='any',
      install_requires=requirements,
      entry_points={
          'console_scripts': [
          ],
      },
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
      ],
)
