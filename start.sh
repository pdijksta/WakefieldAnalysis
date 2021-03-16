#!/bin/bash
sourcefile=/opt/gfa/python
[[ -e $sourcefile ]] && . $sourcefile 3.7
python3 main.py
