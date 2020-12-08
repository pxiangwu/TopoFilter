#!/usr/bin/env bash
g++ -O3 -w -shared -std=c++11 -fPIC -I ../pybind11/include `python3.6-config --cflags --ldflags --libs` pythonGraphTopoFix_withCompInfo.cpp -o PythonGraphPers_withCompInfo.so
cp -p PythonGraphPers_withCompInfo.so ../
