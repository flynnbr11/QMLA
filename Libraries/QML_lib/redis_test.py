import redis 


redis_databases = [
    'test_db_1',
    'test_db_2'
]


def launch_redis_db():
    for i in range(len(redis_databases)):
        db_name = redis_databases[i]
        db_loc = launch_single_db(db_name, i)
        print("launched", db_loc)    
        
        
def launch_single_db(name, db_id):
    db_name = redis.StrictRedis(host="localhost", port=6379, db=db_id)
    return db_name
    

