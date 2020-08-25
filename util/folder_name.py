import datetime
import importlib
import os
from shutil import copyfile
from subprocess import call, check_output
import sys
import time

def get_folder_name(file_name, Config, config_folder):

    layer_str = '_layers_'
    if Config.L1_DIM:
        layer_str += str(Config.L1_DIM)
        if Config.L2_DIM:
            layer_str += '_' + str(Config.L2_DIM)
            if Config.L3_DIM:
                layer_str += '_' + str(Config.L3_DIM)

    out_str = '_lr_' + str(Config.OPT_LR).split('.')[1]

    out_str += '_FC_' + str(Config.FC) + '_dim_' + str(Config.FC_DIM)

    out_str += layer_str

    out_str += '_epochs_' + str(Config.EPOCHS)

    out_str += '_attention_' + str(Config.ATTENTION_FLAG)

    out_str += '_seed_'  + str(Config.SEED_VALUE) + '_'


    start_time = time.time()
    string_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d%H%M%S')

    folder = Config.OUTPUT_FOLDER + file_name + out_str + string_time + '/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Copy config file to output folder
    copyfile(sys.modules[config_folder].__file__, folder + '/config.py')


    return folder

