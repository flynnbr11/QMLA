## Load iqle and qle errors and qlosses into lists to recreate plots. 


%matplotlib inline
plt.clf()
plt.hist(iqle_differences, normed=False, bins=30)
plt.ylabel('Count of IQLE runs');
plt.xlabel('Error');
plt.title('Count of IQLE runs by error margin (single parameter, '+str(num_tests)+' runs)');
plt.savefig('test_plots/iqle_errors_histogram_'+str(num_tests)+'_runs.png')
plt.plot()    


%matplotlib inline
plt.clf()
plt.hist(iqle_final_qloss, normed=False, bins=30)
plt.ylabel('Count of IQLE runs');
plt.xlabel('Error');
plt.title('Count of IQLE runs by final Quadratic Loss (single parameter, '+str(num_tests)+' runs)');
plt.savefig('test_plots/iqle_final_quad_loss_histogram_'+str(num_tests)+'_runs.png')
plt.plot()    
    

%matplotlib inline
plt.clf()
for i in range(num_tests):
    plt.semilogy( range(1, 1+len(iqle_qlosses[i])), iqle_qlosses[i])
plt.xlabel('Experiment Number');
plt.ylabel('Quadratic Loss');
plt.title('IQLE Quadratic Loss per experiment (single parameter, '+str(num_tests)+' runs)');
plt.savefig('test_plots/iqle_scatter_'+str(num_tests)+'_runs.png')
plt.plot()    
    

if just_iqle is not True:    


    %matplotlib inline
    plt.clf()
    plt.hist(qle_differences, normed=False, bins=30)
    plt.ylabel('Count of QLE runs');
    plt.xlabel('Error');
    plt.title('Count of QLE runs by error margin (single parameter, '+str(num_tests)+' runs)');
    plt.savefig('test_plots/qle_errors_histogram_'+str(num_tests)+'_runs.png')
    plt.plot()    



    %matplotlib inline
    plt.clf()
    plt.hist(qle_final_qloss, normed=False, bins=30)
    plt.ylabel('Count of QLE runs');
    plt.xlabel('Error');
    plt.title('Count of QLE runs by final Quadratic Loss (single parameter, '+str(num_tests)+' runs)');
    plt.savefig('test_plots/qle_final_quad_loss_histogram_'+str(num_tests)+'_runs.png')
    plt.plot()    

    num_exps = len(iqle_qlosses[0])
    num_tests= len(iqle_qlosses)



    %matplotlib inline
    plt.clf()
    for i in range(num_tests):
        plt.semilogy( range(1,1+len(qle_qlosses[i])), qle_qlosses[i])
    plt.xlabel('Experiment Number');
    plt.ylabel('Quadratic Loss');
    plt.title('QLE Quadratic Loss per experiment (single parameter, '+str(num_tests)+' runs)');
    plt.savefig('test_plots/qle_scatter_'+str(num_tests)+'_runs.png')
    plt.plot()    



