import redis
import os, sys
#TODO do as list?
#TODO do in function and return unique set of dbs.. or else set list of port ids in QMD, cycle through them so several QMDs can be run simultaneously. 


"""
try:
	host_name = os.getenv("QMD_REDIS_HOST")
except:
	print("Couldn't find environment variable for Redis server name.")
	host_name= 'localhost'


if host_name is None:
    host_name= 'localhost'
#    host_name = 'redis://localhost:6379/1'



print("Using host name ", host_name)
port_number = 6379
"""





test_workers = False
host_name = 'localhost'
port_number = 6379


#use_rq = bool(os.getenv("USE_RQ"))
#print("use rq from environment:", use_rq)

qmd_info_db = redis.StrictRedis(host=host_name, port=port_number, db=0)
learned_models_info = redis.StrictRedis(host=host_name, port=port_number, db=1)
learned_models_ids = redis.StrictRedis(host=host_name, port=port_number, db=2)
bayes_factors_db = redis.StrictRedis(host=host_name, port=port_number, db=3) # Don't think this is a good approach for bayes factors since you'd need an entire db for each model id
bayes_factors_winners_db = redis.StrictRedis(host=host_name, port=port_number, db=4)
active_branches_learning_models = redis.StrictRedis(host=host_name, port=port_number, db=5)
active_branches_bayes = redis.StrictRedis(host=host_name, port=port_number, db=6)
active_interbranch_bayes =  redis.StrictRedis(host=host_name, port=port_number, db=7)





try:
    import pickle
    pickle.HIGHEST_PROTOCOL=2
    from rq import Connection, Queue, Worker

    redis_conn = redis.Redis(host=host_name, port=port_number)
    q = Queue(connection=redis_conn, async=test_workers, default_timeout=3600) # TODO is this timeout sufficient for ALL QMD jobs?
    parallel_enabled = True
except:
    parallel_enabled = False    



def flushdatabases():
    try:
        qmd_info_db.flushdb()
        learned_models_info.flushdb()
        learned_models_ids.flushdb()
        bayes_factors_db.flushdb()
        bayes_factors_winners_db.flushdb()
        active_branches_learning_models.flushdb()
        active_branches_bayes.flushdb()
        active_interbranch_bayes.flushdb()
    except:
        print("Databases don't exist on Redis yet.")
    
def countWorkers():
    # TODO this isn't working
    return Worker.count(connection=redis_conn)
    
    
def hello():
    print("Hello")    
