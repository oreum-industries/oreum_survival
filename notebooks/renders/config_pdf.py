# config_pdf.py
# gets used alongside ~/.jupyter/jupyter_nbconvert_config.py
# which you can create from this dir as `$> jupyter nbconvert --generate-config`
# run this from inside notebooks/ dir:
# $> jupyter nbconvert --config renders/config_pdf.py
c = get_config()
c.NbConvertApp.export_format = "pdf"
c.FilesWriter.build_directory = "renders"  # output-dir
c.NbConvertApp.notebooks = ["000_Intro.ipynb"]
c.PDFExporter.exclude_input_prompt = True
c.PDFExporter.exclude_output_prompt = True
c.PDFExporter.latex_count = 3
c.PDFExporter.verbose = False
c.ExecutePreprocessor.enabled = False
# c.LatexExporter.template_file = 'renders/templates/wip'
