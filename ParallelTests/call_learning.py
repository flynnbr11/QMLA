from learn_model import *
from rq import Connection, Queue
import redis
import pickle

sys.path.append(os.path.join("..", "Libraries","QML_lib"))
import Evo as evo
import DataBase 
from QMD import QMD #  class moved to QMD in Library
import QML
import ModelGeneration 
import BayesF
import matplotlib.pyplot as plt




redis_conn = redis.Redis()
rds = redis.StrictRedis(host='localhost', port=6379, db=0)
q = Queue(connection=redis_conn)


name = 'x'
num_particles = 100
num_exp = 30


q.enqueue(LearnModel, 'x', num_particles=num_particles, num_exp=num_exp)

# LearnModel('x', num_particles=num_particles, num_exp=num_exp)

redis_db_name = str(name)+'_learned'
redis_updater_name = str(name)+'_updater'

p = rds.pubsub()
p.subscribe(redis_db_name)
print("Seeking message from ", redis_db_name)
msg = p.get_message()


print("Message from X : ", msg)

#mod = pickle.loads(rds.get(redis_db_name))
#print("Mod loaded has name : ", mod.Name)
