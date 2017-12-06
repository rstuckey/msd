======================================================
 **MSD** : Mass-Spring-Damper Simulation & Estimation
======================================================

.. image:: https://github.com/stuckeyr/msd/raw/master/msd_estim_output_plot.png
   :align: center
   :alt: Mass-Spring-Damper Estimation Output Plot
   :width: 800px

.. class:: center

Bayesian Model Graph

Requirements
============

The standard Python distribution: http://www.python.org/

The NumPy scientific computing package: http://www.numpy.org/

Matplotlib: http://matplotlib.org/

Cython: http://cython.org/

The Boost C++ libraries: http://www.boost.org/

The odeint C++ library: http://headmyshoulder.github.io/odeint-v2/

PyUblas: http://www.wxpython.org/

Installation
============

Ubuntu Linux
------------

First, download a release version of Boost from: http://www.boost.org/users/download/

Install Boost. I like to keep it local::

  tar xvf boost_X_XX_X.tar.bz2
  cd boost_X_XX_X
  ./bootstrap.sh --with-python=/opt/anaconda3/bin/python3.6 --with-python-version=3.6 --with-python-root=/opt/anaconda3/lib/python3.6
  mkdir ${HOME}/pool
  ./b2 install --with-python --prefix=${HOME}/pool
  cd ..

You may need to edit user-config.jam in $HOME to set Python configuration::

  using python : 3.6 : /opt/anaconda3/bin/python3.6 : /opt/anaconda3/include/python3.6m : /opt/anaconda3/lib/python3.6 ;

Tell the dynamic linker about Boost (add to your .bashrc)::

  export LD_LIBRARY_PATH=${HOME}/pool/lib:${LD_LIBRARY_PATH}

Assuming you have Python installed on your system, make sure you also have the development libraries::

  sudo apt-get install libpython-dev

Clone this repository::

  git clone https://github.com/stuckeyr/msd.git

Either install the required libraries, preferably inside a virtualenv::

  cd msd
  pip install Cython ipython jupyter lmfit matplotlib numpy pymc scipy tqdm
  cd ..

Or install using the requirements::

  cd msd
  pip install -r requirements.txt
  cd ..

Clone the odeint source, and install (copy) into your Boost directory::

  git clone https://github.com/headmyshoulder/odeint-v2.git
  cp -r odeint-v2/include/boost/numeric/ $HOME/pool/include/boost/

Download PyUblas::

  git clone http://git.tiker.net/trees/pyublas.git
  cd pyublas

Create and Customize a Configuration File ".aksetup-defaults.py" in your $HOME directory with the following text::

  BOOST_BINDINGS_INC_DIR = ['${HOME}/pool/include/boost-bindings']
  BOOST_INC_DIR = ['${HOME}/pool/include']
  BOOST_LIB_DIR = ['${HOME}/pool/lib']
  BOOST_PYTHON_LIBNAME = ['boost_python3']

Build PyUblas::

  python setup.py install --user
  cd ..

Install (copy) the include files into your Boost directory::

  cp -r pyublas/pyublas/include/pyublas/ ${HOME}/pool/include/

The instructions to install Pyublas are also here: http://documen.tician.de/pyublas/installing.html

Finally, build the Boost msd model, "msde"::

  cd msd
  python setup-pyublas.py build_ext --inplace

If you encounter a compiler error: "... '_1' was not declared in this scope ...", add the following directive to ${HOME}/pool/include/boost/python/exception_translator.hpp and $HOME/pool/include/boost/python/iterator.hpp, after the include of boost/bind.hpp::

  # include <boost/bind/placeholders.hpp>

Also, expand any reference to _1 and _2 with boost::placeholders::_1 and boost::placeholders::_2, respectively.

If that goes ok, you should have a shared object at msd/msdux*.so

In the same directory build the Cython extension::

  python setup-cython.py build_ext --inplace

And build the Boost extension::

  python setup-boost.py build_ext --inplace

Again, if that goes ok, you should have shared objects at msd/msdc*.so and msd/msdbx*.so

Execution
=========

The best way to run the msd scripts is from within a Jupyter notebook::

  jupyter notebook

You can view the msd notebook here_.

.. _here: http://nbviewer.jupyter.org/github/stuckeyr/msd/blob/master/msd.ipynb

If you want to run the notebook on a separate (local) computer, make sure you set the following in your ".jupyter/jupyter_notebook_config.py" first::

  c.NotebookApp.port = 9999
  c.NotebookApp.ip = '*'
  c.NotebookApp.open_browser = False

In your web browser, go to the host and ip of the computer above.

Select the model to run::

  MODEL = 'boost' # ['python', 'cython', 'boost']

From there, you can start by running the simulation::

  PLOT_SIM = True
  %run -i sim.py

The -i flag retains all variables in the global workspace.

Then try performing a linear regression::

  %run -i reg.py

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
