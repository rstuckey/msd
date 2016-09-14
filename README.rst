======================================================
 **MSD** : Mass-Spring-Damper Simulation & Estimation
======================================================

.. image:: https://bitbucket.org/stuckeyr/bms/raw/master/images/msd-model_graph.png
   :align: center
   :alt: Mass-Spring-Damper Bayesian Model
   :width: 419px

.. class:: center

Bayesian Model Graph

Execution
=========

The best way to run the msd scripts is from within iPython::

  ipython --pylab

Or a Jupyter notebook::

  jupyter notebook

If you want to run the notebook on a separate (local) computer, make sure you set the following in your ".jupyter/jupyter_notebook_config.py" first::

  c.NotebookApp.port = 9999
  c.NotebookApp.ip = '*'
  c.NotebookApp.open_browser = False

In your web browser, go to the host and ip of the computer above.

Select the model to run::

  MODEL = 'boost' # ['python', 'cython', 'boost']

From there, you can start by running the simulation::

  %run -i sim.py

The -i flag retains all variables in the global workspace.

Plot the system response::

  %run -i simplot.py

Then try performing a linear regression::

  %run -i reg.py

And overlay the resulting estimated system response on the original plots::

  %run -i simplot.py

Next, do some iterative. Select the optimisation function::

  OPTFUN = 'lmfit' # ['optimize', 'lmfit']

In order to see the system response from each iteration, set the following global variable::

  PLOT_ESTIM = True

Then perform a nonlinear optimisation::

  %run -i estim.py

Finally, run some Bayesian estimation algorithms::

  %run -i bms.py

And plot some performance parameters::

  %run -i bmsplot.py
