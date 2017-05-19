### Bokeh Linear Regression Example
### Written By: Eric Strong
### Last Updated: 2017/05/10

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as LR
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models.ranges import Range1d
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row, widgetbox
from bokeh.models.widgets import Slider, Button, Div

###-----------------------------------------------------------------------###
###------------------------PARAMETER DEFAULTS-----------------------------###
### This section contains defaults and ranges for the Bokeh controls and  ###
### may be modified without concern, if required. ("View" Part 1)         ###
###-----------------------------------------------------------------------###
# The format for this section is: default, range[Lower, Upper, Step Size]
d_nsamp, r_nsamp = 100, [50, 500, 50] # Number of samples
d_bias, r_bias = 0, [-50, 50, 5] # Bias
d_noise, r_noise = 3, [0, 20, 1] # Amount of noise

###-----------------------------------------------------------------------###
###----------------------GRAPHICAL USER INTERFACE-------------------------###
### This code defines the Bokeh controls that are used for the user       ###
### interface. All the defaults for the controls are above. This code     ###
### should not need to be modified. ("View" Part 2)                       ### 
###-----------------------------------------------------------------------###
# Plot- Regression Data
plot_data = figure(plot_height=300, plot_width=600,
                   title="Regression Data", toolbar_location="above", 
                   x_axis_label='X', y_axis_label='Y',
                   tools="pan,save,box_zoom,wheel_zoom")
# Plot Control Buttons
plot_sim = Button(label="Simulate")
plot_clear = Button(label="Clear")
plot_ctls = column(plot_sim, plot_clear)
# Main Control Buttons
ctl_title = Div(text="<h3>Parameters</h3>")
ctl_nsamp = Slider(title="Number of Samples", value=d_nsamp, 
                  start=r_nsamp[0], end=r_nsamp[1], step=r_nsamp[2])
ctl_bias = Slider(title="Bias", value=d_bias, 
                  start=r_bias[0], end=r_bias[1], step=r_bias[2])
ctl_noise = Slider(title="Noise", value=d_noise, 
                  start=r_noise[0], end=r_noise[1], step=r_noise[2])
ctl_inputs = widgetbox(ctl_title, ctl_nsamp, ctl_bias, ctl_noise)

###-----------------------------------------------------------------------###
###-----------------------BASE-LEVEL FUNCTIONS----------------------------###
### This section contains the low-level calculation for the inspection    ###
### intervals. ("Controller" Part 1)                                      ###
###-----------------------------------------------------------------------###
def create_data(n_samp, bias, noise):
    # Creates a set of random data based on user parameters
    data = make_regression(n_samp, 1, 1, 1, bias, noise=noise)
    return data

def fit_data(data):
    # Uses linear regression to find the coefficients and intercept
    x = data[0]
    y = data[1]
    l_model = LR()
    l_model.fit(x, y)
    return l_model.coef_, l_model.intercept_
    
###-----------------------------------------------------------------------###
###------------------DATA SOURCES AND INITIALIZATION----------------------###
### This section defines the data sources which will be used in the Bokeh ###
### plots. To update a Bokeh plot in the server, each of the sources will ###
### be modified in the CALLBACKS section. ("Model")                       ###
###-----------------------------------------------------------------------###
# Generating some initial data for the plots, based on the parameter defaults
d_data = create_data(d_nsamp, d_bias, d_noise)
d_coef, d_int = fit_data(d_data)
# Find the minimum and maximum values of the data, for ranges, etc.
# d_data[0] is the "X" data, and d_data[1] is the "Y" data
# Remember that the "X" data is of size [n,1] while the "Y" data is [n]
# That's the reason for the extra [0] in the "X" data below
d_x_min = min(d_data[0])[0]
d_x_max = max(d_data[0])[0]
d_y_min = min(d_data[1])
d_y_max = max(d_data[1])
# Find the default regression line
d_x = [d_x_min, d_x_max]
d_y = [d_x_min*d_coef+d_int, d_x_max*d_coef+d_int]
# Defining the Bokeh data sources
source_data = ColumnDataSource(data=dict(x=d_data[0], y=d_data[1]))
source_line = ColumnDataSource(data=dict(x=d_x, y=d_y))
# Associating the sources with plots
plot_data.scatter('x', 'y', source=source_data)
plot_data.line('x', 'y', source=source_line, line_color='orange')
# Defining the plot ranges
xrange_data = Range1d(bounds=[None, None], start=d_x_min, end=d_x_max)
yrange_data = Range1d(bounds=[None, None], start=d_y_min, end=d_y_max)
# Associating the ranges with plots
plot_data.x_range = xrange_data
plot_data.y_range = yrange_data

###-----------------------------------------------------------------------###
###----------------------------CALLBACKS----------------------------------###
### This section defines the behavior of the GUI as the user interacts    ###
### with the controls. ("Controller" Part 2)                              ###
###-----------------------------------------------------------------------###
# Behavior when the "Simulate" button is clicked    
def update_plot():
    # Pull the parameters from the controls
    num_samples = ctl_nsamp.value
    bias = ctl_bias.value
    noise = ctl_noise.value
    # Generate and fit the data
    data = create_data(num_samples, bias, noise)
    coef, inter = fit_data(data)
    # Find the min and maxes of the data
    x_min = min(data[0])[0]
    x_max = max(data[0])[0]
    y_min = min(data[1])
    y_max = max(data[1])
    # Find the regression line
    x = [x_min, x_max]
    y = [x_min*coef+inter, x_max*coef+inter]
    # Update the data sources
    source_data.data = dict(x=data[0], y=data[1])
    source_line.data = dict(x=x, y=y)
    # Update the data ranges
    xrange_data.start = x_min
    xrange_data.end = x_max
    yrange_data.start = y_min
    yrange_data.end = y_max
    
# Behavior when the "Clear" button is clicked
def clear_plot():
    source_data.data = dict(x=[], y=[])

# Button callbacks, using the above functions
plot_sim.on_click(update_plot)
plot_clear.on_click(clear_plot)

###-----------------------------------------------------------------------###
###----------------------------PAGE LAYOUT--------------------------------###
### This section defines the basic layout of the GUI. ("View" Part 3)     ###
###-----------------------------------------------------------------------###
col_inputs = column(plot_ctls, ctl_inputs)
col_plots = column(plot_data)
row_page = row(col_inputs, col_plots, width=1200)
curdoc().add_root(row_page)
curdoc().title = "Linear Regression Simulator"