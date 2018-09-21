from bokeh.plotting import figure
from bokeh.embed import components

import numpy as np

def make_plot():
    
    plot = figure(plot_height=300, sizing_mode='scale_width')



    x = np.arange(1,101)
    y = np.random.randint(0,200,100)
    y2 = np.random.randint(0,200,100)

    print(y2)

    plot.line(x, y, line_width=2,color='#ff69b4')
    plot.line(x, y2, line_width=2,color='#69b4ff')
    

    script, div = components(plot)
    return script, div


#import numpy as np

# def build_plot():

# 	output_file('plot.html',title='Plot')

# 	x_data = np.arange(1,101)
# 	y_data = np.random.randint(0,101,100)

# 	line(x_data,y_data)

# 	snippet = curplot().create_html_snippet(embed_base_url="../static/js",embed_save_loc="../static/js")

# 	return(snippet)