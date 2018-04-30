from __future__ import print_function # so print doesn't show brackets
import redis
import os, sys
import pickle


databases_required  = [
    'qmd_info_db',
    'learned_models_info',
    'learned_models_ids',
    'bayes_factors_db',
    'bayes_factors_winners_db',
    'active_branches_learning_models',
    'active_branches_bayes',
    'active_interbranch_bayes'
]

def databases_from_qmd_id(host_name, port_number, qmd_id, print_status=False):
    database_dict = {}
    seed = get_seed(host_name=host_name, port_number=port_number, qmd_id=qmd_id)
	
    if print_status:
        print("Database requested for host/port/id", host_name, port_number, qmd_id, "has seed", seed)
        qid_seeds = redis.StrictRedis(host=host_name, port=port_number, db=0)
        print("QID seed dict has keys:", qid_seeds.keys())

    for i in range(len(databases_required)):
        new_db = databases_required[i]
        database_dict[new_db] = redis.StrictRedis(host=host_name, port=port_number, db=seed+i)
        
    return database_dict


def get_seed(host_name, port_number, qmd_id, print_status=False):
    #print("Get seed for host", host_name, " port", port_number, "id", qmd_id)
    # db=0 is reserved for a SEED dict: QMD_IDs have a seed
    # their dbs are counted from that seed
    qid_seeds = redis.StrictRedis(host=host_name, port=port_number, db=0)
    
    seed_db_keys = [a.decode() for a in qid_seeds.keys()]
    #print("seed db keys:", seed_db_keys)

    first_qmd_id=False
    
    if 'max' not in seed_db_keys:
        # ie the database has not been set yet
        #print("Max not present; setting")
        qid_seeds.set('max', 1)
        first_qmd_id = True

    if str(qmd_id) in seed_db_keys:
        #print("QMD id", qmd_id, "in", seed_db_keys)
        #print("Returning", int(qid_seeds.get(qmd_id)))
        seed = int(qid_seeds.get(qmd_id))
        
    elif qmd_id not in seed_db_keys:
        max_seed = int(qid_seeds.get('max'))
        if first_qmd_id:
            new_qid_seed = 1
        else:
            new_qid_seed = max_seed+len(databases_required)
        qid_seeds.set(qmd_id, int(new_qid_seed))
        qid_seeds.set('max', new_qid_seed)

        seed = new_qid_seed

    if print_status:
        print("Seed requested for host/port/id", host_name,'/', port_number, '/', qmd_id, ";has seed", seed)
        print("Seed keys", seed_db_keys)
        
    return int(seed)




def flush_dbs_from_id(host_name, port_number, qmd_id):
    dbs = databases_from_qmd_id(host_name, port_number, qmd_id=qmd_id)
    #print("flushing for id ", qmd_id, "host", host_name, "port", port_number, "db=", dbs)
    for v in list(dbs.values()):
        print("flushing",v)
        try:
            v.flushdb()        
        except:
            pass
        
def remove_from_dict(host_name, port_number, qmd_id):
    seeds_dict = redis.StrictRedis(host=host_name, port=port_number, db=0)
    seed = get_seed(host_name, port_number, qmd_id)
    flush_dbs_from_id(host_name, port_number, qmd_id)
    
    del seeds_dict[seed]



def redis_start(host_name, port_number, qmd_id):
    redis_conn = redis.Redis(host=host_name, port=port_number)

    if 'Running' not in redis_conn:
        print("On host/port", host_name, "/", port_number, ": first QMD", qmd_id)
        running_dict = {}
        running_dict[qmd_id] = True
        pickled_running_dict = pickle.dumps(running_dict, protocol=2)
        redis_conn.set('Running', pickled_running_dict)
        
    else:
        print("On host/port", host_name, "/", port_number, ": setting ON QMD", qmd_id)
        current = pickle.loads(redis_conn['Running'])
        current[qmd_id] = True
        pickled_running_dict = pickle.dumps(current, protocol=2)
        redis_conn.set('Running', pickled_running_dict)
        

        
def redis_end(host_name, port_number, qmd_id):
    redis_conn = redis.Redis(host=host_name, port=port_number)
    current = pickle.loads(redis_conn['Running'])
    current[qmd_id] = False
    pickled_running_dict = pickle.dumps(current, protocol=2)
    redis_conn.set('Running', pickled_running_dict)

def check_running(host_name, port_number):
    # Check if all QMD ids on this redis host have finished, ie turned Running to False.
    redis_conn = redis.Redis(host=host_name, port=port_number)
    
    if 'Running' in redis_conn:
        current = pickle.loads(redis_conn['Running'])
        if all(a == False for a in list(current.values())):
            print("On redis host/port", host_name, "/", port_number, ":QMD ids have all finished:", list(current.keys()))
            return 'Finished'
        else:
            return 'Running'

    else:
        return 'Finished'
        
def cleanup_redis(host_name, port_number):
    if check_running(host_name, port_number)=='Finished':
        redis_conn = redis.Redis(host=host_name, port=port_number)
        redis_conn.flushall()
        redis_conn.shutdown()
        print("Redis flushed and shut down on", host_name, "/", port_number)
    else:
        print("Some QMD still active on", host_name, "/", port_number) 
        
