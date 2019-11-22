#!/usr/bin/env bash

# 编译
python setup.py build
# 生成 tar.gz
python setup.py sdist
# 生成 egg 包
python setup.py bdist_egg
# 生成 wheel 包
python setup.py bdist_wheel

#twine register dist/*
# 发布包
twine upload dist/*



rm -rf notekeras.egg-info
rm -rf dist
rm -rf build



git pull
git add -A
git commit -a -m "add"
git push