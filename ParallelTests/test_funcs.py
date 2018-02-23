


def do_test():
    import redis 
    redis_conn = redis.Redis()
    rds = redis.StrictRedis(host='localhost', port=6379, db=0)
#    q = Queue(connection=redis_conn)
    ps = rds.pubsub()
    rds.publish('brians_test_channel', {'this' : 'that', 'a':'b'})
    ps.close()
    print("Testing")
    return "Completed"
    
def repeat_string(this_string='hello'):
    print("Given string is : ", this_string)
    return 1    



    
    
    

