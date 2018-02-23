import time
import pickle
pickle.HIGHEST_PROTOCOL = 2
from rq import Connection, Queue, Worker
import redis
from test_funcs import * # added import!

redis_conn = redis.Redis()
q = Queue(connection=redis_conn)
rds = redis.StrictRedis(host='localhost', port=6379, db=0)
ps = rds.pubsub()




ps.subscribe('brians_test_channel')

print("Number of workers : ", Worker.count(connection=redis_conn))

for i in range(10):
  b = q.enqueue(do_test ,result_ttl=-1) # result won't expire but must be deleted later
  a = b.return_value
  print("a = " , a)
  time.sleep(0.5)
  
  
  
"""
for i in range(20):
    message = ps.get_message()
    if message: 
        print("Received message : ", message['data'])
        
        print("Total dict received : ", message)
        
    time.sleep(1)
"""    
