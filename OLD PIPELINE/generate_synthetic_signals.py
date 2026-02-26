import os
import sys
import math
import yaml
import numpy as np
import argparse
sys.path.insert(0,os.path.abspath('../../config_file_handling/'))
from config_file_handling import get_intermediate_data_file_name

#default run config file
default_run_definition="../../config/run1c_definitions.yaml"
target_nibble="test_nibble"


argparser=argparse.ArgumentParser()
argparser.add_argument("-r","--run_definition",help="run definition yaml file",default=default_run_definition)
argparser.add_argument("-n","--nibble_name",help="name of nibble to run",default=target_nibble)
argparser.add_argument("-s","--synthetic",help="synthetic arrangement name")
args=argparser.parse_args()

#pull in run configuration
run_definition_file=open(args.run_definition,"r")
run_definition=yaml.load(run_definition_file)
run_definition_file.close()
target_nibble=args.nibble_name
target_synth=args.synthetic

start_frequency=float(run_definition["nibbles"][target_nibble]["start_freq"])
stop_frequency=float(run_definition["nibbles"][target_nibble]["stop_freq"])
candidate_density=run_definition["software_synthetics"][target_synth]["density_per_MHz"]
ksvz_power_ratio=run_definition["software_synthetics"][target_synth]["ksvz_power_ratio"]
lineshape=run_definition["software_synthetics"][target_synth]["lineshape"]
eavg_hz=run_definition["software_synthetics"][target_synth]["eavg_hz"]
synth_file_name=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"synthetics.yaml")

print("synth_file_name", synth_file_name)

#start_frequency=5100
#stop_frequency=5800
#number=100
number=math.floor((stop_frequency-start_frequency)*candidate_density/1e6)

frequencies=np.linspace(start_frequency,stop_frequency,number)
#ksvz_power_ratio=1e6
#eavg_hz=1755

signals=[]
for f in frequencies:
    signals.append( {"frequency": float(f),"ksvz_power_ratio": ksvz_power_ratio,"eavg_hz": eavg_hz,"lineshape": lineshape} )

with open(synth_file_name,"w") as yaml_file:
    yaml.dump({"signals": signals},yaml_file)
    yaml_file.close()

#print(yaml.dump({"signals": signals}))
