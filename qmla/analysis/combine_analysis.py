import os
import qmla.get_exploration_strategy as get_exploration_strategy

def combine_analysis_plots(
    results_directory,
    output_file_name,
    variables,
    include_all_pngs=False
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
            self.cell(30, 10, analysis_title, 0, 0, 'C')
            # Line break
            self.ln(20)

        # Page footer
        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            # Arial italic 8
            self.set_font('Arial', 'I', 8)
            # Page number
            self.cell(0, 10, 'Page ' +
                      str(self.page_no()) +
                      '/{nb}', 0, 0, 'C')

    # Instantiation of inherited class
    pdf = PDF()

    run_dir = re.search(
        # 'results/(.*)/',
        'results/(.*)',
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
        run_directory=run_dir
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
    pdf.set_text_color(r=0, g=0, b=0)  # reset to black

    if include_all_pngs:
        image_list = [
            str(results_directory + a)
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
