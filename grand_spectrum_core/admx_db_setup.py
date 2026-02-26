import grand_spectrum_core.admx_db_interface
import argparse
import yaml
import sys
import os
from grand_spectrum_core.admx_db_datatypes import PowerSpectrum
sys.path.insert(0,os.path.abspath('../../config_file_handling/main_analysis/'))
from grand_spectrum_core.config_file_handling import get_intermediate_data_file_name

class ADMXSetup:
    "initialization that happens in every python file"

    def __init__(self):
        #default run config file                                                                                      
        print("initialized")
    
    def get_initialization_params(self,output_file_name_string):
        #default run config file                                                                                                                  
        default_run_definition="run1b_definitions.yaml"
        target_nibble="nibble5"

        argparser=argparse.ArgumentParser()
        argparser.add_argument("-r","--run_definition",help="run definition yaml file",default=default_run_definition)
        argparser.add_argument("-n","--nibble_name",help="name of nibble to run",default=target_nibble)
        argparser.add_argument("-u","--update",help="update this file (not yet implemented)",action="store_true")
        argparser.add_argument("-s","--synthetic",help="synthetic signal file")
        argparser.add_argument("-x","--x_nibble",help="does nibble have two start/stop times?",action='store_true')
        args=argparser.parse_args()
        
        # run definition
        run_definition_file=open(args.run_definition,"r")
        run_definition=yaml.safe_load(run_definition_file)
        run_definition_file.close()
        target_nibble=args.nibble_name

        db=admx_db_interface.ADMXDB()
        #db.hostname="localhost"
        db.hostname="localhost"
        db.dbname=run_definition["database"]
        if "database_port" in run_definition:
            db.port=run_definition["database_port"]
        is_sidecar=False
        if "channel" in run_definition:
            if run_definition["channel"]=="sidecar":
                print("SIDECAR MODE")
                is_sidecar=True

        print("hostname is {}".format(db.hostname))
        print("dbname is {}".format(db.dbname))
        print("port is {}".format(db.port))
        print("is sidecar {}".format(is_sidecar))

        output_file=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],output_file_name_string)

        synthetic_signals=[]
        if args.synthetic:
            synth_file=open(args.synthetic,"r")
            synth_data=yaml.load(synth_file)
            synthetic_signals=synth_data["signals"]
            synth_file.close()
            print("{} synthetic signals loaded".format(len(synthetic_signals)))

        return db,run_definition, target_nibble, is_sidecar, output_file, synthetic_signals, args.update, args.x_nibble

