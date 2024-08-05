# Plotting

## Resources

Read or watch:

- [Plot (graphics)](https://en.wikipedia.org/wiki/Plot_(graphics))
- [Scatter plot](https://en.wikipedia.org/wiki/Scatter_plot)
- [Line chart](https://en.wikipedia.org/wiki/Line_chart)
- [Bar chart](https://en.wikipedia.org/wiki/Bar_chart)
- [Histogram](https://en.wikipedia.org/wiki/Histogram)
- [Pyplot tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
- [matplotlib.pyplot](https://matplotlib.org/stable/api/pyplot_api.html)

## Learning Objectives

By the end of this project, you should be able to explain:

- What is a plot?
- What is a scatter plot, line graph, bar graph, histogram?
- What is matplotlib?
- How to plot data with matplotlib
- How to label a plot
- How to scale an axis
- How to plot multiple sets of data simultaneously

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All files will be interpreted/compiled on Ubuntu 20.04 LTS using `python3` (version 3.9)
- Your files will be executed with `numpy` (version 1.25.2) and `matplotlib` (version 3.8.3)
- All files should end with a new line
- The first line of all your files should be `#!/usr/bin/env python3`
- A `README.md` file at the root of the project folder is mandatory
- Your code should use the `pycodestyle` style (version 2.11.1)
- All modules should have documentation
- All classes should have documentation
- All functions (inside and outside a class) should have documentation
- You are not allowed to import any module unless noted otherwise
- All files must be executable
- File length will be tested using `wc`

## More Info

### Installing Matplotlib

```sh
pip install --user matplotlib==3.8.3
pip install --user Pillow==10.2.0
sudo apt-get install python3-tk

To check the installation, use pip list.

Configure X11 Forwarding
Update your Vagrantfile:
Vagrant.configure(2) do |config|
  ...
  config.ssh.forward_x11 = true
end
Tasks
Line Graph

Filename: 0-line.py
Plot y as a solid red line from 0 to 10.
Scatter Plot

Filename: 1-scatter.py
Plot x ↦ y as a scatter plot with magenta points.
Change of Scale

Filename: 2-change_scale.py
Plot x ↦ y as a line graph with logarithmic y-axis.
Two is Better Than One

Filename: 3-two.py
Plot x ↦ y1 and x ↦ y2 as line graphs.
Frequency

Filename: 4-frequency.py
Plot a histogram of student scores for a project.
All in One

Filename: 5-all_in_one.py
Plot all 5 previous graphs in one figure.
Stacking Bars

Filename: 6-bars.py
Plot a stacked bar graph of the number of fruit various people possess.
