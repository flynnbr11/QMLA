import sys
import redis
from rq import Queue, Connection, Worker
import argparse

#Preload libraries
import qmla


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
# redis_conn = redis.Redis(
#         host = redis_host_name,
#         port = redis_port_number
# )

# with Connection( redis_conn ):

#         w = Worker( [str(qmla_id)] , connection=redis_conn)
#         w.work()