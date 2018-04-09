import sys, os
import pickle
import matplotlib.pyplot as plt
import argparse


def model_scores(directory_name):
#    sys.path.append(directory_name)

    print("current:", os.getcwd())
    os.chdir(directory_name)

    scores = {}

    pickled_files = []
    for file in os.listdir(directory_name):
        if file.endswith(".p") and file.startswith("results"):
            pickled_files.append(file)
    
    for f in pickled_files:
        fname = directory_name+'/'+str(f)
        result = pickle.load(open(fname, 'rb'))
        alph = result['NameAlphabetical']

        if alph in scores.keys():
            scores[alph] += 1
        else:
            scores[alph] = 1
    return scores
    

def plot_scores(scores, save_file='model_scores.png'):
    plt.clf()
    models = list(scores.keys())
    scores = list(scores.values())
    
    plt.title('Number of QMD instances won by models')
    plt.xlabel('Model')
    plt.ylabel('Number of wins')
    plt.bar(models,scores)
    
    plt.savefig(save_file)
    

parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')

# Add parser arguments, ie command line arguments for QMD
## QMD parameters -- fundamentals such as number of particles etc
parser.add_argument(
  '-dir', '--results_directory', 
  help="Directory where results of multiple QMD are held.",
  type=str,
  default=os.getcwd()
)


arguments = parser.parse_args()
directory_to_analyse = arguments.results_directory
plot_file = directory_to_analyse+'/model_scores.png'
print("directory : ", directory_to_analyse)

model_scores = model_scores(directory_to_analyse)
plot_scores(model_scores, plot_file)

