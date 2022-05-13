# IBM Analog Hardware Acceleration Kit: Unit Tests


This project contains a unittest compatible test suite, that can be executed by any Python test runner. The recommended runner is pytest, which can be installed along with a number of other development tools used in aihwkit via:
```
pip install -r requirements-dev.txt 
```

To run the full test suite simply do (from the command line):
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
