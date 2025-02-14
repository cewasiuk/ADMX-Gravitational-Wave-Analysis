import os
import re

import sys
import math
import datetime
import yaml
import pytz
from dateutil import parser

#---have to reset the way timestamps are handeled to be timezone aware---
#def timestamp_constructor(loader,node):
#    return parser.parse(node.value)
#yaml.add_constructor(u'tag:yaml.org,2002:timestamp', timestamp_constructor)
#---------

def frequency_cuts(frequency_cut_yaml, target_time, target_frequency):
    for cut_def in frequency_cut_yaml:
        seattle_tz = pytz.timezone("US/Pacific")
        if cut_def["stop_time"].tzinfo is None:
            start_time = seattle_tz.localize(cut_def["start_time"])
            stop_time = seattle_tz.localize(cut_def["stop_time"])
        else:
            start_time = cut_def["start_time"]
            stop_time = cut_def["stop_time"]
        if target_time>start_time and target_time<stop_time and target_frequency>float(cut_def["start_frequency"]) and target_frequency<float(cut_def["stop_frequency"]):
           return True,"frequency_cut: "+cut_def["why"]
    return False,""

def test_timestamp_cuts(timestamp_cut_yaml,target_time):
    seattle_tz = pytz.timezone('US/Pacific')
    if target_time.tzinfo is None:
        target_time=seattle_tz.localize(target_time)
    for cut_def in timestamp_cut_yaml:
        if cut_def["stop_time"].tzinfo is None:
            start_time = seattle_tz.localize(cut_def["start_time"])
            stop_time = seattle_tz.localize(cut_def["stop_time"])
        else:
            start_time = cut_def["start_time"]
            stop_time = cut_def["stop_time"]
        if target_time>start_time and target_time<stop_time: 
            return True,"timestamp_cut: "+cut_def["why"]
    return False,""

def shell_variable_replace(path):
    return re.sub(r'(?<!\\)\$[A-Za-z_][A-Za-z0-9_]*', '', os.path.expandvars(path))

def get_intermediate_data_file_name(config_yaml,label):
    data_dir = config_yaml["data_directory"]
    full_data_dir = data_dir+"/"+config_yaml["file_prefix"]+"_"+label
    return full_data_dir
    #return shell_variable_replace(config_yaml["data_directory"])+"/"+config_yaml["file_prefix"]+"_"+label

