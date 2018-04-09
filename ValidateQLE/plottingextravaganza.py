import numpy as np

def ising_name_processing(name):
    terms=name.split('PP')
    rotations = ['xTi', 'yTi', 'zTi']
    hartree_fock = ['xTx', 'yTy', 'zTz']
    transverse = ['xTy', 'xTz', 'yTz']
    
    
    present_r = []
    present_hf = []
    present_t = []
    
    for t in terms:
        if t in rotations:
            present_r.append(t[0])
        elif t in hartree_fock:
            present_hf.append(t[0])
        elif t in transverse:
            string = t[0]+t[-1]
            present_t.append(string)
        else:
            print("Term",t,"doesn't belong to rotations, Hartree-Fock or transverse.")

    present_r.sort()
    present_hf.sort()
    present_t.sort()
    
    return present_r, present_hf, present_t
    

def latex_processing(splitname):    
    outstring = ['$']
    indices = range(len(splitname))
    for termindex in indices:
        termclass = splitname[termindex]
        if len(termclass)>0:
            if termindex is 0 :
                outstring.append('R_{')
            elif termindex is 1 :
                outstring.append('HF_{')
            else:
                outstring.append('T_{')
            for term in termclass:
                outstring.append(term)
                if (termindex is 2) and (termclass.index(term) is not len(termclass)-1):
                    outstring.append(',')
            outstring.append('}')
    outstring.append('$')

    return r''.join(outstring)
    
    
def BayF_IndexDictToMatrix(ModelNames, AllBayesFactors, StartBayesFactors=None):
    
    size = len(ModelNames)
    Bayf_matrix = np.zeros([size,size])
    
    for i in range(size):
        for j in range(size):
            if j > i:
                try: 
                    Bayf_matrix[i,j] = AllBayesFactors[i][j]
                except:
                    Bayf_matrix[i,j] = 1
    
            elif j<i and (StartBayesFactors is not None):
                try: 
                    Bayf_matrix[i,j] = StartBayesFactors[i][j]
                except:
                    Bayf_matrix[i,j] = 1
    
    return Bayf_matrix