# -*- coding: utf-8 -*-

# (C) Copyright 2020 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

.PHONY: build_inplace clean clang-tidy clang-format mypy pycodestyle pylint pytest

build_inplace:
	 python setup.py build_ext -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE --inplace

clean:
	python setup.py clean
	cd docs && make clean
	rm -f src/aihwkit/simulator/rpu_base.*.so

clang-format:
	clang-format -i src/aihwkit/simulator/rpu_base_src/*.cpp  \
	src/aihwkit/simulator/rpu_base_src/*.h \
	src/rpucuda/*.cpp src/rpucuda/*.h \
	src/rpucuda/cuda/*.cpp src/rpucuda/cuda/*.h src/rpucuda/cuda/*.cu

doc:
	cd docs && make html

mypy:
	mypy --show-error-codes src/

pycodestyle:
	pycodestyle src/ tests/

pylint:
	PYTHONPATH=src/ pylint -rn src/ tests/

pytest:
	PYTHONPATH=src/ pytest -v -s tests/
