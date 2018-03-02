import redis

#TODO do as list?
qmd_info_db = redis.StrictRedis(host="localhost", port=6379, db=0)
learned_models_info = redis.StrictRedis(host="localhost", port=6379, db=1)
learned_models_ids = redis.StrictRedis(host="localhost", port=6379, db=2)
bayes_factors_db = redis.StrictRedis(host="localhost", port=6379, db=3) # Don't think this is a good approach for bayes factors since you'd need an entire db for each model id

try:
    import pickle
    pickle.HIGHEST_PROTOCOL=2
    from rq import Connection, Queue, Worker

    redis_conn = redis.Redis()
    q = Queue(connection=redis_conn)
    parallel_enabled = True
except:
    parallel_enabled = False    



def flushdatabases():
    qmd_info_db.flushdb()
    learned_models_info.flushdb()
    learned_models_ids.flushdb()
    bayes_factors_db.flushdb()
    
    
def countWorkers():
    # TODO this isn't working
    return Worker.count(connection=redis_conn)
