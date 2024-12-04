# config_slides.py
# gets used alongside ~/.jupyter/jupyter_nbconvert_config.py
# which you can create from this dir as `$> jupyter nbconvert --generate-config`
# run this from inside notebooks/ dir:
# $> jupyter nbconvert --config renders/config_slides.py
# default reveal themes: https://revealjs.com/themes/
# insufficient docs https://nbconvert.readthedocs.io/en/latest/customizing.html
# general themeing
# https://nbconvert.readthedocs.io/en/latest/customizing.html
# https://github.com/jelleschutter/nbconvert-theme-pale-sand-navy
# https://github.com/sheeshee/nbconvert-template-example
c = get_config()

c.NbConvertApp.export_format = "slides"
c.FilesWriter.build_directory = "renders"  # output-dir
c.NbConvertApp.notebooks = ["000_Intro.ipynb"]
c.SlidesExporter.embed_images = True  # True! otherwise uses stupid image links
c.SlidesExporter.exclude_input = True
c.SlidesExporter.exclude_input_prompt = True
c.SlidesExporter.exclude_output_prompt = True
c.SlidesExporter.reveal_number = "h"  # 'c/t'
c.SlidesExporter.reveal_width = "1600"  #'1280'  # 1920
c.SlidesExporter.reveal_height = "900"  # '720'  #'1080'
c.SlidesExporter.reveal_scroll = False
c.SlidesExporter.reveal_theme = "simple"  #'serif'
c.SlidesExporter.reveal_transition = "slide"  # 'slide' # none fade convex concave zoom
c.SlidesExporter.skip_svg_encoding = True
# c.SlidesExporter.extra_template_basedirs = 'renders/templates/'
# c.SlidesExporter.template_name = 'slides-oemplus'
