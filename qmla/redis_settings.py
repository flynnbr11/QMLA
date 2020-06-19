import os
import sys
import pickle

import redis

r"""
Create multiple databases on a redis server.

Redis connections require a port number and host server.
Each QMLA instance has a unique port number (though can share hosts).
These are determined by the launch script,
either local_launch.sh or parallel_launch.sh.
By calling get_redis_databases_by_qmla_id, a set of database
addresses unique to that qmla_id (host/port) are returned in a dictionary.
These can be used to store information required by distinct actors
to perform QMLA, without requiring all information be held on all actors
at all times.
This method is quite slow - useful information is stored in dictionaries and
pickled to redis. Pickling and unpickling is quite slow, so should be minimised.
"""

__all__ = [
    'databases_required',
    'get_redis_databases_by_qmla_id',
    'get_seed'
]


databases_required = [
    'qmla_core_info_database',
    'learned_models_info_db',
    'learned_models_ids',
    'bayes_factors_db',
    'bayes_factors_winners_db',
    'active_branches_learning_models',
    'active_branches_bayes',
    'active_interbranch_bayes',  # TODO unused?
    'any_job_failed'
]


def get_redis_databases_by_qmla_id(
    host_name,
    port_number,
    qmla_id,
    tree_identifiers=None,
):
    r"""
    Gets the set of redis databases unique to this QMLA instance.

    Each :class:`~qmla.QuantumModelLearningAgent` instance is associated with a unique
    redis database. Redis databases are specified by their `hostname` and `port number`.
    All workers for the QMLA instance can read the redis databsae of that instance.
    Data required by various workers is stored here, through
    :meth:`~qmla.QuantumModelLearningAgent._compile_and_store_qmla_info_summary`.

    A set of databases are stored at the redis database host_name:port_number;
    these are listed in ``qmla.redis_settings.databases_required``.

    :param str host_name: name of host server on which redis database exists.
    :param int port_number: this QMLA instance's unique port number (6300 + qmla_id).
    :param int qmla_id: QMLA id, unique to a single instance within a run.
    :return dict database_dict: set of database addresses unique to the qmla_id, host_name and port_number.
    """

    database_dict = {}
    # Seed this QMLA instance's database ID's
    seed = get_seed(
        host_name=host_name,
        port_number=port_number,
        qmla_id=qmla_id)
    # TODO is seed always 0 since port=6300+qmla_id? possibly remove if so,
    # doesn't provide extra protection

    for i in range(len(databases_required)):
        # name the new database by the listing in databases_required
        new_db = databases_required[i]
        # place a new database for this data set on the redis database
        database_dict[new_db] = redis.StrictRedis(
            host=host_name,
            port=port_number,
            db=seed + i
        )

    return database_dict


def get_seed(host_name, port_number, qmla_id):
    r"""
    Unique seed for this QMLA id.

    Numerous databases can belong to a given host:port address,
    and these are identified by their ``db`` attribute (a number to keep
    databases separate).
    Databases are seeded using the ``qmla_id``, as well as the host:port,
    to avoid multiple QMLA instances, which can exist on the same host and port,
    clashing and interfering with each others' data.

    E.g.
    * A host:port is already in use for ``qmla_id=1``, which uses a set of 5 databases.
    * ``qmla_id=2`` requests a set of databases on the same host:port
    * The first available ``db`` is 6, such that ``qmla_id=2`` will not interfere with
      the databases of ``qmla_id=1``.

    :param str host_name: name of host server on which redis database exists.
    :param int port_number: this QMLA instance's unique port number (6300 + qmla_id).
    :param int qmla_id: QMLA id, unique to a single instance within a run.
    :return int seed: unique number to use as the starting ``db`` for a given QMLA
        instances set of databases.
    """

    qid_seeds = redis.StrictRedis(host=host_name, port=port_number, db=0)
    seed_db_keys = [a.decode() for a in qid_seeds.keys()]
    first_qmla_id = False

    if 'max' not in seed_db_keys:
        # ie the database has not been set yet
        qid_seeds.set('max', 1)
        first_qmla_id = True

    if str(qmla_id) in seed_db_keys:
        seed = int(qid_seeds.get(qmla_id))

    elif qmla_id not in seed_db_keys:
        max_seed = int(qid_seeds.get('max'))
        if first_qmla_id:
            new_qid_seed = 1
        else:
            new_qid_seed = max_seed + len(databases_required)
        qid_seeds.set(qmla_id, int(new_qid_seed))
        qid_seeds.set('max', new_qid_seed)

        seed = new_qid_seed
    return int(seed)
