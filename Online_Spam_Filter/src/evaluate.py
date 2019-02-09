#!/usr/bin/env python
# -*- coding:utf-8 -*-
from Feature_extraction import *
from functions import *

test_label = load_class_data("../data/derived/test_label.txt")
classify_results = load_class_data("../data/derived/classify_results.txt")
evaluate(classify_results, test_label)