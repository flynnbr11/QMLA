import argparse
import os
import GrowthRules

global test_growth_class_implementation
test_growth_class_implementation = True

class store_variables():
    def __init__(self, **kwargs):
        self.variables = {}
        for k in kwargs:
            self.variables[str(k)] =  kwargs[k]

def combine_analysis_plots(
    results_directory,
    output_file_name,
    variables
):
    import re
    from fpdf import FPDF
    class PDF(FPDF):

        def set_descriptors(
            self, 
            run_directory=None, 
        ):
            self.run_directory = run_directory


        def header(self):
            # Logo
    #         self.image('logo_pb.png', 10, 8, 33)
            # Arial bold 15
            self.set_font('Arial', 'B', 15)
            # Move to the right
            self.cell(80)
            # Title
            analysis_title = str(
                'QMD Analysis [{}]'.format(
                    self.run_directory
                ) 
            )
            self.cell(30, 10, analysis_title , 0, 0, 'C' )
            # Line break
            self.ln(20)

        # Page footer
        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            # Arial italic 8
            self.set_font('Arial', 'I', 8)
            # Page number
            self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    # Instantiation of inherited class
    pdf = PDF()
    
    run_dir = re.search(
        # 'Results/(.*)/',
        'Results/(.*)',
        results_directory
    ).group(1)
    
    output_desc = str(
        run_dir.split('/')[0] + 
        '__' + 
        run_dir.split('/')[1]
    )
    if not results_directory.endswith('/'):
        results_directory += '/'

    pdf.set_descriptors(
        run_directory = run_dir 
    )
    pdf.alias_nb_pages()
    pdf.add_page()


    pdf.set_font('Times', '', 12)
    for k in variables.keys():
        pdf.set_text_color(r=255, g=0, b=0)
        pdf.write(
            8, 
            str(str(k) + ' : ')
        )
        pdf.set_text_color(r=0, g=0, b=0)
        pdf.write(
            8, 
            str(str(variables[k]) + '\n')
        )
    pdf.set_text_color(r=0, g=0, b=0) # reset to black

    image_list = [
        str(results_directory  + a) 
        for a in os.listdir(results_directory) 
        if '.png' in a
    ]

    for new_img in image_list:
        pdf.add_page()
        pdf.image(
            new_img, 
            w=200
        )
    output_file_name = str(
        output_desc + '_' + output_file_name
    )

    output_file = str(
        results_directory + 
        output_file_name
    )
    print("Saving to :", output_file)
    pdf.output(output_file, 'F')

######    

parser = argparse.ArgumentParser(
    description='Pass files to pickel QHL parameters.'
)

parser.add_argument(
  '-dir', '--results_directory',
  help='Absolute path to directory to store results in.',
  type=str,
  default=''
)

parser.add_argument(
  '-out', '--output_file_name',
  help='Absolute path to directory to store results in.',
  type=str,
  default='analysis.pdf'
)



parser.add_argument(
  '-e', '--num_experiments', 
  help='Number of experiments to use for the learning process',
  type=int,
  default=0
)
parser.add_argument(
  '-p', '--num_particles', 
  help='Number of particles to use for the learning process',
  type=int,
  default=0
)
parser.add_argument(
  '-bt', '--num_times_bayes', 
  help='Number of times to consider in Bayes function.',
  type=int,
  default=0
)
parser.add_argument(
  '-nprobes', '--num_probes',
  help='How many probe states in rota for learning parameters.',
  type=int,
  default=20
)

parser.add_argument(
  '-pnoise', '--probe_noise_level',
  help='Noise level to add to probe for learning',
  type=float,
  default=0.03
)
parser.add_argument(
  '-special_probe', '--special_probe_for_learning',
  help='Specify type of probe to use during learning.',
  type=str,
  default=None
)
parser.add_argument(
  '-ggr', '--growth_generation_rule',
  help='Rule applied for generation of new models during QMD. \
    Corresponding functions must be built into ModelGeneration',
  type=str,
  default='Unknown'
)
parser.add_argument(
  '-run_desc', '--run_description',
  help='Short description of this run',
  type=str,
  default='Unknown'
)
parser.add_argument(
  '-git_commit', '--git_commit_hash',
  help='Hash of git commit',
  type=str,
  default=''
)
parser.add_argument(
  '-t', '--num_tests', 
  help="Number of complete tests to average over.",
  type=int,
  default=1
)
parser.add_argument(
  '-rt', '--resample_threshold',
  help='Resampling threshold for QInfer.',
  type=float,
  default=0.5
)
parser.add_argument(
  '-ra', '--resample_a',
  help='Resampling a for QInfer.',
  type=float,
  default=0.98
)
parser.add_argument(
  '-pgh', '--pgh_factor',
  help='Resampling threshold for QInfer.',
  type=float,
  default=1.0
)
parser.add_argument(
  '-qhl', '--qhl_test', 
  help="Bool to test QHL on given true operator only.",
  type=int,
  default=0
)
parser.add_argument(
  '-mqhl', '--multiQHL',
  help='Run QHL test on multiple (provided) models.',
  type=int,
  default=0
)
parser.add_argument(
  '-cb', '--cumulative_csv',
  help='CSV to store Bayes factors of all QMDs.',
  type=str,
  default='Unknown'
)

parser.add_argument(
  '-exp', '--experimental_data',
  help='Use experimental data if provided',
  type=int,
  default=0
)

# import UserFunctions

arguments = parser.parse_args()
results_directory = arguments.results_directory
output_file_name = arguments.output_file_name

growth_generation_rule = arguments.growth_generation_rule
growth_class = GrowthRules.get_growth_generator_class(
  growth_generation_rule = growth_generation_rule,
  use_experimental_data = arguments.experimental_data
)

variables = vars(arguments)
# and some others arguments not explicitly set in launch script

# variables['measurement_type'] = growth_class.measurement_type
variables['expectation_value_func'] = growth_class.expectation_value_function.__name__
variables['heuristic'] = growth_class.heuristic_function.__name__
variables['probe_generation_function'] = growth_class.probe_generation_function.__name__
variables['plot_probe_generation_function'] = growth_class.plot_probe_generation_function.__name__

combine_analysis_plots(
    results_directory = results_directory,
    output_file_name = output_file_name,
    variables = variables
)