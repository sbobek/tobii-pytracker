Contributing
============

Contributions to Tobii-PyTracker are welcome, including bug reports, feature
requests, documentation improvements, tests, and code contributions.

Reporting issues
----------------

Before opening a new issue, check whether a similar issue already exists. Use
the appropriate GitHub issue template and provide enough information to
reproduce and diagnose the problem. For bug reports, include:

* a clear description of the problem;
* steps or a minimal example required to reproduce it;
* the expected and observed behavior;
* the Tobii-PyTracker and Python versions;
* the operating system and runtime environment, such as native Linux, Windows,
  Docker, WSL, or a virtual machine;
* the eye-tracker model, mouse emulator, or information that eye tracking was
  disabled;
* relevant package versions; and
* the complete error message, traceback, or relevant logs, if applicable.

Do not include participant data, recordings, credentials, confidential data,
or machine-specific secrets in issues, examples, screenshots, or logs.

Contributing code
-----------------

To contribute code:

#. Fork the repository.
#. Create a branch for your changes.
#. Implement and test your changes.
#. Update the documentation when necessary.
#. Submit a pull request using the provided pull request template.

Keep each pull request focused on a single issue or feature where possible.
Describe the motivation for the change, its implementation, and how it was
tested. If the change is hardware- or platform-dependent, state the environment
in which it was validated.

Code style
----------

Python code should follow `PEP 8 <https://peps.python.org/pep-0008/>`_
conventions.

New or modified code should:

* use clear and descriptive names;
* include docstrings for public modules, classes, methods, and functions;
* remain consistent with the style of the existing codebase;
* avoid unrelated formatting or refactoring changes;
* include or update tests where appropriate; and
* remain readable and maintainable across the supported runtime environments.

Testing
-------

The project currently supports Python 3.10. From the repository root, install
the project and its test dependencies, then run the test suite:

.. code-block:: console

   python -m pip install -e .
   python -m pip install -r tests/requirements-test.txt
   python -m pytest tests/ -v

Ensure that existing tests continue to pass before submitting a pull request.
CI runs the tests on Ubuntu with a virtual display. Changes affecting display,
audio, eye-tracker connectivity, gaze collection, or timing may also require
validation in the relevant runtime environment or on appropriate hardware.
Describe any such validation in the pull request. If hardware-dependent testing
is not available, state this limitation explicitly.

Documentation
-------------

Changes that modify installation, configuration, command-line options, public
APIs, examples, recorded-data formats, or other user-visible behavior should be
accompanied by corresponding documentation updates where appropriate. Examples
should be concise, reproducible, and should not contain participant or
confidential data.

Pull requests
-------------

A pull request should include:

* a concise description of the change;
* a link to the related issue, when applicable;
* a summary of the tests performed;
* information about the operating system, runtime, emulator, or eye-tracker
  used for validation, when relevant; and
* documentation updates, when applicable.

By submitting a contribution, you agree that it may be distributed under the
license used by the Tobii-PyTracker project.

Authors
-------

* Szymon Bobek <szymon.bobek@gmail.com>
* Dmytro Polishchuk <dmitro.polischuk@gmail.com>
* Sebastian Sęczyk <sebastian.seczyk@gmail.com>
* Jakub Pleśniak <jakubpl2001@gmail.com>
