CUR_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.SILENT:

prepare:
	pip install nvidia-cutlass-dsl nvmath-python
	echo " -- Preparing environment --"
	echo "export PYTHONPATH=\$$PYTHONPATH:$(CUR_DIR)"
	

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

NSYS_OUTPUT_NAME := gb200
nsys:
	ONLY_PROFILE=1 nsys profile --gpu-metrics-devices=all -f true \
	-o $(NSYS_OUTPUT_NAME) pytest -s -v tests/pytorch/test_linear_cross_entropy.py


FWD_OUTPUT_NAME := gb200_fwd_mainloop
ncu-fwd:
	ONLY_PROFILE=1 ncu --kernel-name regex:fwd_mainloop \
		--launch-skip 1 --launch-count 1 --set full --clock-control=none \
		-f -o $(FWD_OUTPUT_NAME) \
		pytest -s -v tests/pytorch/test_linear_cross_entropy.py

ncu-fwd-cli:
	ONLY_PROFILE=1 ncu --kernel-name regex:fwd_mainloop \
		--launch-skip 1 --launch-count 1 --clock-control=none \
		--log-file fwd_mainloop.log \
		pytest -s -v tests/pytorch/test_linear_cross_entropy.py

BWD_OUTPUT_NAME := gb200_bwd_partial_dlogits
ncu-bwd:
	ONLY_PROFILE=1 ncu --kernel-name regex:bwd_partial_dlogits \
		--launch-skip 133 --launch-count 1 \
		--set full --clock-control=none \
		-f -o $(BWD_OUTPUT_NAME) \
		pytest -s -v tests/pytorch/test_linear_cross_entropy.py

ncu-bwd-cli:
	ONLY_PROFILE=1 ncu --kernel-name regex:bwd_partial_dlogits \
		--launch-skip 133 --launch-count 1 \
		--clock-control=none \
		--log-file bwd_partial_dlogits.log \
		pytest -s -v tests/pytorch/test_linear_cross_entropy.py