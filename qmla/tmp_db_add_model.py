def launch_db(
    # true_op_name,
    new_model_branches,
    new_model_ids,
    log_file,
    RootN_Qbit=[0],
    N_Qubits=1,
    gen_list=[],
    true_model_terms_matrices=[],
    true_model_terms_params=[],
    num_particles=1000,
    qle=True,
    redimensionalise=True,
    resample_threshold=0.5,
    resampler_a=0.95,
    pgh_prefactor=1.0,
    num_probes=None,
    probe_dict=None,
    use_exp_custom=True,
    enable_sparse=True,
    debug_directory=None,
    qid=0,
    host_name='localhost',
    port_number=6379,
    **kwargs
):
    """
    Inputs:
    TODO

    Outputs:
      - db: "running database", info on active models. Can access QML and
        operator instances for all information about models.
      - legacy_db: when a model is terminated, we maintain essential information
        in this db (for plotting, debugging etc.).
      - model_lists = list of lists containing alphabetised model names.
        When a new model is considered, it

    Usage:
        $ gen_list = ['xTy, yPz, iTxTTy] # Sample list of model names
        $ running_db, legacy_db, model_lists = database_framework.launch_db(gen_list=gen_list)

    """

    Max_N_Qubits = 13
    model_lists = {}
    for j in range(1, Max_N_Qubits):
        model_lists[j] = []

    legacy_db = pd.DataFrame({
        '<Name>': [],
        'Param_Est_Final': [],
        'Epoch_Start': [],
        'Epoch_Finish': [],
        'ModelID': [],
    })

    db = pd.DataFrame({
        '<Name>': [],
        'Status': [],  # TODO can get rid?
        'Completed': [],  # TODO what's this for?
        'branch_id': [],  # TODO proper branch id's,
        # 'Param_Estimates' : sim_ops,
        # 'Estimates_Dist_Width' : [normal_dist_width for gen in generators],
        # 'Model_Class_Instance' : [],
        'Reduced_Model_Class_Instance': [],
        'Operator_Instance': [],
        'Epoch_Start': [],
        'ModelID': [],
    })

    model_id = int(0)

    gen_list = list(new_model_branches.keys())

    for model_name in gen_list:
        try_add_model = add_model(
            model_name=model_name,
            model_id=int(new_model_ids[model_name]),
            branch_id=new_model_branches[model_name],
            running_database=db,
            model_lists=model_lists,
            true_model_terms_matrices=true_model_terms_matrices,
            true_model_terms_params=true_model_terms_params,
            log_file=log_file,
            qid=qid,
            host_name=host_name,
            port_number=port_number
            # true_op_name=true_op_name,
            # epoch=0,
            # probe_dict=probe_dict,
            # resample_threshold=resample_threshold,
            # resampler_a=resampler_a,
            # pgh_prefactor=pgh_prefactor,
            # num_probes=num_probes,
            # num_particles=num_particles,
            # redimensionalise=redimensionalise,
            # use_exp_custom=use_exp_custom,
            # enable_sparse=enable_sparse,
            # debug_directory=debug_directory,
            # branch_id=0,
            qle=qle,
        )
        if try_add_model is True:
            model_id += int(1)

    return db, legacy_db, model_lists
