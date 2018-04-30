import RedisSettings as rds
import argparse
import redis 

parser = argparse.ArgumentParser(description='Redis settings.')

parser.add_argument(
  '-rh', '--redis_host_name', 
  help="Redis host.",
  type=str,
  default='localhost'
)

parser.add_argument(
  '-rpn', '--redis_port_number', 
  help="Redis port number.",
  type=int,
  default=6379
)

parser.add_argument(
  '-rqid', '--redis_qmd_id', 
  help="QMD ID.",
  type=int,
  default=0
)

parser.add_argument(
  '-action', '--redis_action', 
  help="QMD ID.",
  type=str,
  default=0
)


arguments = parser.parse_args()

host_name = arguments.redis_host_name
port_number = arguments.redis_port_number
qmd_id = arguments.redis_qmd_id
action  = arguments.redis_action


if action=='add':
    rds.redis_start(host_name, port_number, qmd_id)
elif action=='remove':
    rds.redis_end(host_name, port_number, qmd_id)
else:
    print("Redis Manager: action should be either 'add' or 'remove'")
