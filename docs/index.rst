.. tobii-pytracker documentation master file, created by
   sphinx-quickstart on Fri Feb 23 10:20:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Tobii-Pytracker documentation
-----------------------------------------------

**Tobii-Pytracker** is a Python-based framework for Tobii eye tracker experiments and multimodal data collection.

.. image:: https://raw.githubusercontent.com/sbobek/tobii-pytracker/refs/heads/psychopy/pix/tobii-pytracker.svg
    :width: 240px
    :align: center
    :alt: Tobii-Pytracker Workflow

Install
=======

Tobii-Pytracker can be installed from either `PyPI <https://pypi.org/project/tobii-pytracker>`_ or directly from source code `GitHub <https://github.com/sbobek/tobii-pytracker>`_

To install form PyPI::

   pip install tobii-pytracker

To install from source code::

   git clone https://github.com/sbobek/tobii-pytracker
   cd tobii-pytracker
   pip install .
   pip install psychopy>=2024.1.4 --no-deps

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   Basic Usage examples <basic_examples>

.. toctree::
   :maxdepth: 2
   :caption: Reference

   API reference <api>

.. toctree::
   :maxdepth: 1
   :caption: Development

   Troubleshooting <troubleshooting>
   Release notes <release_notes>
   Contributing guide <contributing>

