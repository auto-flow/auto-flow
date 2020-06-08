#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os
import re
import sys

from setuptools import setup, find_packages

with open("autoflow/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()


if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: %s. '
    )

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. AutoFlow requires Python '
        '3.6 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

with open('README.rst') as fh:
    long_description = fh.read()

GIT_PATTERN = re.compile(r"git\+https://github\.com/(.*?)/(.*?)\.git")
HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'requirements.txt')) as fp:
    install_requires = []
    for r in fp.readlines():
        r = r.strip()
        if r.startswith("#"):
            continue
        elif r.startswith("git+"):
            match = GIT_PATTERN.match(r)
            if match:
                package_name = match.group(2)
            install_require = f"{package_name} @ {r}"
            install_requires += [install_require]
        else:
            install_requires += [r]


def get_package_data(name, suffixes):
    g = os.walk(name)
    lst = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            lst.append(os.path.join(path, file_name))
    ret = []
    for item in lst:
        for suffix in suffixes:
            if item.endswith(suffix):
                ret.append(item)
                break
    M = len(name) + 1
    ret = [s[M:] for s in ret]
    return ret


needed_suffixes = ['.json', '.txt', '.yml', '.yaml', '.bz2', '.csv']

setup(
    name='auto-flow',
    version=version,
    author='qichun tang',
    author_email='tqichun@gmail.com',
    description='AutoFlow: automatic machine learning workflow modeling platform.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    license='BSD',
    url='https://github.com/auto-flow/autoflow',
    packages=find_packages("./", exclude=['test', 'examples', 'autoflow_server']),
    package_dir={
        'autoflow': './autoflow',
        'dsmac': './dsmac',
        'generic_fs': './generic_fs'
    },
    package_data={'autoflow': get_package_data('autoflow', needed_suffixes),
                  'dsmac': get_package_data('dsmac', needed_suffixes)},
    python_requires='>=3.6.*',
    install_requires=install_requires,
    platforms=['Linux'],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
