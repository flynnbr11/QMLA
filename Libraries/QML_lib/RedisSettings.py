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

def databases_from_qmd_id(host_name, port_number, qmd_id):
    database_dict = {}
    seed = get_seed(host_name=host_name, port_number=port_number, qmd_id=qmd_id)
    print("Database requested for host/port/id", host_name, port_number, qmd_id, "has seed", seed)

    for i in range(len(databases_required)):
        new_db = databases_required[i]
        database_dict[new_db] = redis.StrictRedis(host=host_name, port=port_number, db=seed+i)
        
    return database_dict


def get_seed(host_name, port_number, qmd_id):
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
        return int(qid_seeds.get(qmd_id))
        
    elif qmd_id not in seed_db_keys:
        max_seed = int(qid_seeds.get('max'))
        if first_qmd_id:
            new_qid_seed = 1
        else:
            new_qid_seed = max_seed+len(databases_required)
        qid_seeds.set(qmd_id, int(new_qid_seed))
        qid_seeds.set('max', new_qid_seed)
        #print("Adding QMD_id", qmd_id, "to Redis server on host", host_name, ", port", port_number)
        
        #print("for host, port, id", host_name, port_number, qmd_id, "not in", seed_db_keys)
        return new_qid_seed

    return database_dict  

def flush_dbs_from_id(host_name, port_number, qmd_id):
    dbs = databases_from_qmd_id(host_name, port_number, qmd_id=qmd_id)
    #print("flushing for id ", qmd_id, "host", host_name, "port", port_number, "db=", dbs)
    for v in list(dbs.values()):
        v.flushdb()        
