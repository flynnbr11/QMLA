Tutorial
========

Here we provide a complete example of how to run the framework,
including how to implement a custom , and generate/interpret analysis.

Installation 
------------
First, *fork* the :term:`QMLA` codebase from
[QMLA]_ to a Github user account (referred to
as in [listing:qmla\_setup]). Now, we must download the code base and
ensure it runs properly; these instructions are implemented via the
command line.

The steps of preparing the codebase are (i) install ; (ii) create a
virtual Python environment for installing :term:`QMLA` dependencies
without damaging other parts of the user’s environment; (iii) download
the :term:`QMLA` codebase from the forked Github repository; (iv)
install packages upon which :term:`QMLA` depends.

.. code-block:: shell
    :name: qmla_setup

    # Install redis (database broker)
    sudo apt update
    sudo apt install redis-server
     
    # make directory for QMLA
    cd
    mkdir qmla_test
    cd qmla_test

    # make Python virtual environment for QMLA
    # note: change Python3.6 to desired version
    sudo apt-get install python3.6-venv 
    python3.6 -m venv qmla-env    
    source qmla-env/bin/activate

    # Download QMLA
    git clone --depth 1 https://github.com/username/QMLA.git # REPLACE username

    # Install dependencies
    cd QMLA 
    pip install -r requirements.txt 

Note there may be a problem with some packages in the arising from the
attempt to install them all through a single call to :code:`requirements.txt`. 
Ensure these are all installed before proceeding.
Test reference :ref:`qmla_setup`. 
When all of the requirements are installed, test that the framework
runs. :term:`QMLA` uses databases to store intermittent data: we must
manually initialise the database. Run the following (note: here we list
, but this must be corrected to reflect the version installed on the
user’s machine in the above setup section):

::

    ~/redis-4.0.8/src/redis-server

which should give something like [fig:terminal\_redis].

.. figure:: appendix/figures/terminal_redis.png
   :alt: Terminal running .
   :width: 90.0%

   Terminal running .
