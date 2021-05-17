#!/bin/bash
#sourcefile=/opt/gfa/python
#[[ -e $sourcefile ]] && . $sourcefile 3.7
#export RPN_DEFNS=/afs/psi.ch/user/d/dijkstal_p/bin/defns.rpn
#export PYTHONPATH=/afs/psi.ch/user/d/dijkstal_p/pythonpath
[[ $(hostname) == "pc11292.psi.ch" ]] && export PYTHONPATH=$PYTHONPATH:~/pythonpath2/pyqt5
python3 main.py
