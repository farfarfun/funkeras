#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/03/30 12:10
# @Author  : niuliangtao
# @Site    :
# @File    : setup.py
# @Software: PyCharm

from setuptools import setup, find_packages

install_requires = []

setup(name='notekeras',
      version='0.0.2',
      description='notekeras',
      author='niuliangtao',
      author_email='1007530194@qq.com',
      url='https://github.com/1007530194',

      packages=find_packages(),
      install_requires=install_requires
      )
