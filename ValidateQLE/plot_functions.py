
def get_directory_name_by_time(just_date=False):
    import datetime
    # Directory name based on date and time it was generated 
    # from https://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/
    now =  datetime.date.today()
    year = now.strftime("%y")
    month = now.strftime("%b")
    day = now.strftime("%d")
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    date = str (str(day)+'_'+str(month)+'_'+str(year) )
    time = str(str(hour)+'_'+str(minute))
    name = str(date+'/'+time+'/')
    if just_date is False:
        return name
    else: 
        return str(date+'/')



#####################################

### Plotting functions 

#######################################


def store_data_for_plotting(iqle_qle):
    #qlosses=[]
    if iqle_qle == 'qle':
        qlosses = qle_qlosses
        final_qlosses = qle_final_qloss
        qle_type = 'QLE'
        differences = qle_differences
    elif iqle_qle == 'iqle':
        qlosses = iqle_qlosses
        final_qlosses = iqle_final_qloss
        qle_type = 'IQLE'
        differences = iqle_differences

    else:
        print("Needs to either be QLE or IQLE")
        
        
    ### Format data to be used in plotting
        
    exp_values = {}
    for i in range(num_exp):
        exp_values[i] = []

    for i in range(num_tests): 
        for j in range(len(qlosses[i])):
            exp_values[j].append(qlosses[i][j])

    medians=[]        
    std_dev=[]
    means=[]
    mins=[]
    maxs = []
    for k in range(num_exp):
        medians.append(np.median(exp_values[k]) )
        means.append(np.mean(exp_values[k]) )
        mins.append(np.min(exp_values[k]) )
        maxs.append(np.max(exp_values[k]) )
        std_dev.append(np.std(exp_values[k]) )
    
    if iqle_qle == 'qle':
        qle_all_medians.append(medians)
        qle_all_means.append(means)
        qle_all_mins.append(mins)
        qle_all_maxs.append(maxs)
        qle_all_std_devs.append(std_dev)
        qle_descriptions_of_runs.append(description)
    elif iqle_qle == 'iqle':
        iqle_all_medians.append(medians)
        iqle_all_means.append(means)
        iqle_all_mins.append(mins)
        iqle_all_maxs.append(maxs)
        iqle_all_std_devs.append(std_dev)
        iqle_descriptions_of_runs.append(description)
        
#    old_descriptions_of_runs[len(old_descriptions_of_runs)] = description
    
    
def individual_plots(iqle_qle):            
    if iqle_qle == 'qle':
        qlosses = qle_qlosses
        final_qlosses = qle_final_qloss
        qle_type = 'QLE'
        differences = qle_differences
        medians = qle_all_medians
        means = qle_all_means
        mins = qle_all_mins
        maxs = qle_all_maxs
        std_devs = qle_all_std_devs

        
    elif iqle_qle == 'iqle':
        qlosses = iqle_qlosses
        final_qlosses = iqle_final_qloss
        qle_type = 'IQLE'
        differences = iqle_differences
        medians = iqle_all_medians
        means = iqle_all_means
        mins = iqle_all_mins
        maxs = iqle_all_maxs
        std_devs = iqle_all_std_devs
    else:
        print("Needs to either be QLE or IQLE")

    ##### Plots #####
    ### Overall results
    
    # Errors histogram
    # %matplotlib inline
    plot_description = 'Errors_histogram'
    plt.clf()
    plt.hist(differences, normed=False, bins=30)
    plt.ylabel('Count of '+ qle_type +' runs');
    plt.xlabel('Error');
    plt.title('Count of '+ qle_type +' runs by error margin '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)
    
    # Quadratic Loss histogram
    # %matplotlib inline
    plot_description='Final_quadratic_loss'
    plt.clf()
    plt.hist(final_qlosses, normed=False, bins=30)
    plt.ylabel('Count of ' +qle_type+' runs');
    plt.xlabel('Error');
    plt.title('Count of ' +qle_type+' runs by final Quadratic Loss '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)

    # Quadratic loss development of all tests 
    # %matplotlib inline
    plot_description='All_quadratic_loss'
    plt.clf()
    for i in range(num_tests):
        plt.semilogy( range(1,1+len(qlosses[i])), list(qlosses[i]))
    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type+' Quadratic Loss per experiment '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)
    
    
    
            
    ### Averages
    
    """
    # Median Quadratic loss with error bar
    # Not useful here but kept for use case of errorbar function
    # %matplotlib inline
    plot_description = 'Median_with_error_bar'
    plt.clf()
    y = medians
    x = range(1,1+num_exp)
    err = std_dev
    fig,ax = plt.subplots()
    ax.errorbar(x, y, yerr=err)
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')
    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Median Quadratic Loss '+title_appendix);
    plt.savefig(plot_directory+qle_type+'_'+ plot_description)
    """
    # Median with shadowing
    # %matplotlib inline
    plot_description = 'Median_with_std_dev'
    plt.clf()
    #plt.axes.Axes.set_yscale('log')
    y = medians[0]
    x = range(1,1+num_exp)
    #print("Different length lists")
    devs = std_devs[0]
    y_lower = [( y[i] - devs[i]) for i in range(len(y))]
    y_upper = [ (y[i] + devs[i] )for i in range(len(y))]
    #y_lower = np.array(y[0])  - np.array(std_devs[0]) 
    #y_upper = np.array(y[0])  + np.array(std_devs[0])
    fig,ax = plt.subplots()
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')

    plt.plot(x,y)
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, linewidth=0)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Median Quadratic Loss '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)
    
    
    # Median with shadowing
    # %matplotlib inline
    plot_description = 'Median_min_max'
    plt.clf()
    #plt.axes.Axes.set_yscale('log')
    y = medians[0]
    
    x = range(1,1+num_exp)
    y_lower = maxs[0]
    y_upper = mins[0]
    
    fig,ax = plt.subplots()
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')

    plt.plot(x,y)
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, linewidth=0)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Median Quadratic Loss (Max/Min) '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)
        
    
    # Mean with shadowing
    # %matplotlib inline
    plot_description = 'Mean_with_std_dev'
    plt.clf()
    #plt.axes.Axes.set_yscale('log')
    y = means[0]
    devs =  std_devs[0]
    x = range(1,1+num_exp)
    y_lower = [( y[i] - devs[i]) for i in range(len(y))]
    y_upper = [ (y[i] + devs[i] )for i in range(len(y))]
    
    fig,ax = plt.subplots()
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')

    plt.plot(x,y)
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, linewidth=0)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Mean Quadratic Loss '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)

    # Mean with shadowing for min/max
    # %matplotlib inline
    plot_description = 'Mean_min_max'
    plt.clf()
    #plt.axes.Axes.set_yscale('log')
    y = means[0]
    x = range(1,1+num_exp)
    y_lower = mins[0]
    y_upper = maxs[0]
    
    fig,ax = plt.subplots()
    ax.set_yscale('log', nonposy="clip")
    ax.set_xscale('linear')

    plt.plot(x,y)
    plt.fill_between(x, y_lower, y_upper, alpha=0.2, linewidth=0)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type + ' Mean Quadratic Loss (Max/Min) '+title_appendix);
    if save_figs: plt.savefig(plot_directory+qle_type+'_'+ plot_description)

    
def draw_summary_plots(iqle_qle):
    if iqle_qle == 'qle':
        qlosses = qle_qlosses
        final_qlosses = qle_final_qloss
        qle_type = 'QLE'
        differences = qle_differences
        all_qlosses = all_qle_final_qlosses
        descriptions_of_runs = qle_descriptions_of_runs
        all_medians = qle_all_medians
        all_means = qle_all_means
        all_mins = qle_all_mins
        all_maxs = qle_all_maxs
        all_std_devs = qle_all_std_devs
        
    elif iqle_qle == 'iqle':
        qlosses = iqle_qlosses
        final_qlosses = iqle_final_qloss
        qle_type = 'IQLE'
        differences = iqle_differences
        all_qlosses = all_iqle_final_qlosses
        descriptions_of_runs = iqle_descriptions_of_runs
        all_medians = iqle_all_medians
        all_means = iqle_all_means
        all_mins = iqle_all_mins
        all_maxs = iqle_all_maxs
        all_std_devs = iqle_all_std_devs
    else:
        print("Needs to either be QLE or IQLE")

    # Correlation diagram between Quadratic Loss and distance of true param from center of prior    
    distance_from_center = [ item - 0.5 for item in all_true_params_single_list]
    # %matplotlib inline
    plot_description = 'Correlation_distance_quad_loss_'+variable_parameter
    plt.clf()
    plt.xlabel('Distance from center of Prior distribution')
    plt.ylabel('Final Quadratic Loss')
    plt.gca().set_yscale('log')
    x = distance_from_center
    y = all_qlosses
    plt.scatter(x,y)        
    plt.title(qle_type + ' Correlation between distance from center and final quadratic loss');
    plt.savefig(summary_plot_directory+qle_type+'_'+ plot_description)

    
    # Median Quadratic loss with variable resampling threshold
    # %matplotlib inline
    plot_description='Median_Q_Loss_w_variable_'+variable_parameter
    plt.clf()
    for i in range(len(all_medians)):
        plt.semilogy( range(1,1+(num_exp)), list(all_medians[i]), label=descriptions_of_runs[i])

    ax = plt.subplot(111)
    ## Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ## Put a legend below current axis
    lgd=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=4)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type+' Median Quadratic Loss with variable '+ variable_parameter);
    plt.savefig(summary_plot_directory+qle_type+'_'+ plot_description, bbox_extra_artists=(lgd,), bbox_inches='tight')


    # Mean Quadratic loss with variable resampling threshold
    # %matplotlib inline
    plot_description='Mean_Q_Loss_w_variable_'+variable_parameter
    plt.clf()
    for i in range(len(all_medians)):
        plt.semilogy( range(1,1+(num_exp)), list(all_means[i]), label=descriptions_of_runs[i])
    ax = plt.subplot(111)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    lgd=ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=4)

    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title(qle_type+' Mean Quadratic Loss with variable '+ variable_parameter);
    plt.savefig(summary_plot_directory+qle_type+'_'+ plot_description, bbox_extra_artists=(lgd,), bbox_inches='tight')


