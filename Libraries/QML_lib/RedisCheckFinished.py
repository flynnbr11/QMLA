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

arguments = parser.parse_args()

host_name = arguments.redis_host_name
port_number = arguments.redis_port_number
qmd_id = arguments.redis_qmd_id


try:
    running = rds.check_running(host_name, port_number, print_status=False)
    if running == 'Running':
        print("redis-running")
    elif running == 'Finished':
        print("redis-finished")
    else:
        print("Problem with check_running function in RedisManageServer")

except redis.ConnectionError:
    print("redis-finished")
