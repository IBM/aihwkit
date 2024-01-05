# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

.PHONY: build_inplace clean clean-doc clang-format mypy pycodestyle pylint pytest build_inplace_mkl build_inplace_cuda build_cuda

build_inplace:
	python setup.py build_ext -j8 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE --inplace ${flags}

build_inplace_mkl:
	make build_inplace flags="-DRPU_BLAS=MKL -DINTEL_MKL_DIR=${MKLROOT} ${flags}"

build:
	python setup.py install --user -j8 -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE  ${flags}

build_mkl:
	make build flags="-DRPU_BLAS=MKL -DINTEL_MKL_DIR=${MKLROOT}  ${flags}"

build_cuda:
	make build_mkl flags="-DUSE_CUDA=ON ${flags}"

build_inplace_cuda:
	make build_inplace_mkl flags="-DUSE_CUDA=ON ${flags}"

clean:
	python setup.py clean
	rm -rf _skbuild
	rm -f src/aihwkit/simulator/rpu_base*.so
	rm -f src/aihwkit/extension/*.so

clean-doc:
	cd docs && make clean

clang-format:
	git ls-files | grep -E ".*\.(cu|h|cpp)$$" | xargs clang-format -i

doc:
	cd docs && make html

mypy:
	mypy --show-error-codes src/

pycodestyle:
	pycodestyle src/ tests/ examples/

pylint:
	PYTHONPATH=src/ git ls-files | grep -E ".*\.py$$" | grep -v "pb2\.py$$" | xargs  pylint -rn

pytest:
	PYTHONPATH=src/ pytest -v -s tests/

black:
	git ls-files | grep \.py$$ | xargs black -t py310 -C --config .black
