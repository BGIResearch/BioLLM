#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/17 14:50
# @Author  : qiuping
# @File    : run_gears.py
# @Email: qiupinghust@163.com

"""
change log: 
    2024/6/17 14:50  create file. 
"""
from biollm.task.perturbation.gears_task import GearsTask


import sys

config_file = sys.argv[1]
obj = GearsTask(config_file)
obj.run()
