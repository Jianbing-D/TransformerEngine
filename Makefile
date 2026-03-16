CUR_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.SILENT:

prepare:
	export PYTHONPATH=$$PYTHONPATH:$(CUR_DIR)
	pip install nvidia-cutlass-dsl nvmath-python

unit-test-1gpu:
	timeout 3m pytest -s -v tests/pytorch/test_linear_cross_entropy.py

unit-test-4gpu:
	timeout 3m torchrun --nproc_per_node=4 --nnodes 1 -m pytest -s -v tests/pytorch/test_linear_cross_entropy.py

stats-1gpu:
	# Run single-GPU performance and storage tests; show [INFO] stats lines
	timeout 3m pytest -s -v tests/pytorch/test_linear_cross_entropy.py::TestFusedLinearCrossEntropyDataParallel::test_performance tests/pytorch/test_linear_cross_entropy.py::TestFusedLinearCrossEntropyDataParallel::test_storage 2>&1 | grep -E "\[INFO\]:|PASSED|FAILED"

stats-4gpu:
	# Run multi-GPU (TensorParallel + SequenceParallel) performance and storage tests; show [INFO] stats
	timeout 3m torchrun --nproc_per_node=4 --nnodes 1 -m pytest -s -v \
		tests/pytorch/test_linear_cross_entropy.py::TestFusedLinearCrossEntropyTensorParallel::test_performance \
		tests/pytorch/test_linear_cross_entropy.py::TestFusedLinearCrossEntropyTensorParallel::test_storage \
		tests/pytorch/test_linear_cross_entropy.py::TestFusedLinearCrossEntropySequenceParallel::test_performance \
		tests/pytorch/test_linear_cross_entropy.py::TestFusedLinearCrossEntropySequenceParallel::test_storage \
		2>&1 | grep -E "\[INFO\]:|PASSED|FAILED"

OUTPUT_NAME := gb200_fwd_mainloop
ncu:
	ONLY_PROFILE=1 ncu --kernel-name regex:fwd_mainloop \
		--launch-skip 1 --launch-count 1 --set full --clock-control=none \
		-f -o $(OUTPUT_NAME) \
		pytest -s -v tests/pytorch/test_linear_cross_entropy.py