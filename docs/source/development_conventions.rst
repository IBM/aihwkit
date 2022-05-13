Development conventions
=======================

`aihwkit` is an open source project. This section describes how we organize
the work and the conventions and procedures we use for developing the library.

Code conventions
----------------

In order to keep the codebase consistent and assist us in spotting bugs and
issues, we use different tools:

* Python:

  * |pycodestyle|_: for ensuring that we conform to PEP-8, as the minimal
    common style standard.
  * |pylint|_: for being able to identify common pitfalls and potential issues
    in the code, along with additional style conventions.
  * |mypy|_: for taking advantage of type hints and be able to identify issues
    before runtime and help maintenance.

* C++:

  * |clang-format|_: for providing a unified style to the C++ sources. Note
    that different versions result in slightly different output - please use
    the ``10.x`` versions.

* Testing:

  * |pytest|_: while we strive for keeping the project tests stdlib compatible,
    we encourage using ``pytest`` as the test runner for its advanced features.

For convenience, a ``Makefile`` is provided in the project, in order to invoke
the different tools easily. For example::

    make pycodestyle
    make pylint
    make mypy
    make clang-format

Continuous integration
----------------------

The project uses continuous integration: when a new pull request is made or
updated, the different tools and the tests will automatically be run under
different environments (different Python versions, operative systems).

We rely on the result of those checks to help reviewing pull requests: when
contributing, please make sure of reviewing the result of the continuous
integration in order to help fixing potential issues.

Testing
-------

The following environment variables can be used for selecting the subset of
tests to be performed:

* ``TEST_DATASET``: if set, the tests that involve using datasets will be
  executed (involves downloading the datasets).
* ``TEST_CREATE``: if set, the tests that involve remote experiment creation
  will be executed.

The aihwkit project contains a unittest compatible test suite, that can be executed by any Python test runner. 
The recommended runner is pytest, which can be installed along with a number of other development tools used in 
aihwkit via:
```
pip install -r requirements-dev.txt 
```

To run the full test suite simply run from the command line:
```
make pytest
```

To run individual test files, you can use, e.g.
```
pytest -v -s tests/test_presets.py
```

Individual tests can be run by giving the name, e.g.:
```
pytest -v -s tests/test_presets.py::PresetTest_TTv2IdealizedPreset::test_tile_preset
```


Branches and releases
---------------------

For the branches organization:

* the ``master`` branch contains the latest changes and updates. We strive for
  keeping the branch runnable and working, but its contents can be considered
  experimental and "bleeding edge".

When the time for a new release comes:

  * a new ``git tag`` is created. This tag can be used for referencing to that
    stable version of the codebase.
  * a new package is published on PyPI.

This package uses semantic versioning for the version numbers, albeit with
an extra part as we are under beta. For a version number ``0.MAJOR.MINOR``, we
strive to:

1. MAJOR number will be increased when we make incompatible API changes.
2. MINOR number will be increased when we add functionality that is backwards
   compatible, or backwards compatible bug fixes.

Please be aware that during the initial development rounds, there are cases
where we might not be able to adhere fully to the convention.


.. |pycodestyle| replace:: ``pycodestyle``
.. _`pycodestyle`: https://github.com/PyCQA/pycodestyle
.. |pylint| replace:: ``pylint``
.. _`pylint`: https://www.pylint.org/
.. |mypy| replace:: ``mypy``
.. _`mypy`: https://mypy-lang.org/
.. |clang-format| replace:: ``clang-format``
.. _`clang-format`: https://clang.llvm.org/docs/ClangFormat.html
.. |pytest| replace:: ``pytest``
.. _`pytest`: https://pytest.org/
