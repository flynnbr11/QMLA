import sys
import redis
from rq import Queue, Connection, Worker
import argparse

#Preload libraries
# import qmla

import copy
import numpy as np
import time
import pickle
import random
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import redis
import qinfer

import qmla
# import qmla.remote_bayes_factor
# import qmla.remote_model_learning
# import qmla.database_framework as database_framework
# import qmla.model_instances as QML
# import qmla.model_for_comparison
# import qmla.redis_settings as rds
# import qmla.logging

# Parse command line arguments

parser = argparse.ArgumentParser(description='Pass variables for QMLA.')

parser.add_argument(
        '-host', '--host_name',
        help="Redis host's name.",
        type=str,
        default='localhost'
)

parser.add_argument(
        '-port', '--port_number',
        help="Redis port number.",
        type=int,
        default=6379
)
parser.add_argument(
        '-qid', '--qmla_id',
        help='ID tag for QMD.',
        type=int,
        default=1
)
arguments = parser.parse_args()
redis_host_name = arguments.host_name
redis_port_number = arguments.port_number
qmla_id = arguments.qmla_id

print("Custom RQ script. Host:{}; port:{}.".format(redis_host_name, redis_port_number))

# # make a redis connection
redis_conn = redis.Redis(
        host = redis_host_name,
        port = redis_port_number
)

with Connection( redis_conn ):

        w = Worker( [str(qmla_id)] , connection=redis_conn)
        w.work()