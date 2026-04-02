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

entropy-test-1gpu:
	timeout 3m pytest -s -v tests/pytorch/test_linear_cross_entropy_with_entropy.py

entropy-test-4gpu:
	timeout 3m torchrun --nproc_per_node=4 --nnodes 1 -m pytest -s -v tests/pytorch/test_linear_cross_entropy_with_entropy.py

entropy-stats-1gpu:
	# Run single-GPU entropy performance and storage tests; show [INFO] stats lines
	timeout 3m pytest -s -v \
		tests/pytorch/test_linear_cross_entropy_with_entropy.py::TestLinearCrossEntropyWithEntropyDataParallel::test_performance \
		tests/pytorch/test_linear_cross_entropy_with_entropy.py::TestLinearCrossEntropyWithEntropyDataParallel::test_storage \
		2>&1 | grep -E "\[INFO\]:|PASSED|FAILED"

entropy-stats-4gpu:
	# Run multi-GPU entropy (TP + SP) performance and storage tests; show [INFO] stats
	timeout 3m torchrun --nproc_per_node=4 --nnodes 1 -m pytest -s -v \
		tests/pytorch/test_linear_cross_entropy_with_entropy.py::TestLinearCrossEntropyWithEntropyTensorParallel::test_performance \
		tests/pytorch/test_linear_cross_entropy_with_entropy.py::TestLinearCrossEntropyWithEntropyTensorParallel::test_storage \
		tests/pytorch/test_linear_cross_entropy_with_entropy.py::TestLinearCrossEntropyWithEntropySequenceParallel::test_performance \
		tests/pytorch/test_linear_cross_entropy_with_entropy.py::TestLinearCrossEntropyWithEntropySequenceParallel::test_storage \
		2>&1 | grep -E "\[INFO\]:|PASSED|FAILED"

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
		--launch-skip 2 --launch-count 1 \
		--clock-control=none \
		--log-file bwd_partial_dlogits.log \
		pytest -s -v tests/pytorch/test_linear_cross_entropy.py

ncu-bwd-cublas-cli:
	ONLY_PROFILE=1 ncu --kernel-name nvjet_sm100_tss_128x256_64x6_2x1_2cta_v_badd_NNT \
		--launch-skip 129 --launch-count 1 \
		--clock-control=none \
		--log-file bwd_partial_dlogits_cublas.log \
		pytest -s -v tests/pytorch/test_linear_cross_entropy.py


# --- Entropy NCU profiling targets ---

ENTROPY_NSYS_OUTPUT_NAME := gb200_entropy
entropy-nsys:
	ONLY_PROFILE=1 nsys profile --gpu-metrics-devices=all -f true \
	-o $(ENTROPY_NSYS_OUTPUT_NAME) pytest -s -v tests/pytorch/test_linear_cross_entropy_with_entropy.py

ENTROPY_FWD_OUTPUT_NAME := gb200_fwd_mainloop_entropy
entropy-ncu-fwd:
	ONLY_PROFILE=1 ncu --kernel-name regex:fwd_mainloop_entropy \
		--launch-skip 1 --launch-count 1 --set full --clock-control=none \
		-f -o $(ENTROPY_FWD_OUTPUT_NAME) \
		pytest -s -v tests/pytorch/test_linear_cross_entropy_with_entropy.py

entropy-ncu-fwd-cli:
	ONLY_PROFILE=1 ncu --kernel-name regex:fwd_mainloop_entropy \
		--launch-skip 1 --launch-count 1 --clock-control=none \
		--log-file fwd_mainloop_entropy.log \
		pytest -s -v tests/pytorch/test_linear_cross_entropy_with_entropy.py

ENTROPY_BWD_OUTPUT_NAME := gb200_bwd_partial_dlogits_entropy
entropy-ncu-bwd:
	ONLY_PROFILE=1 ncu --kernel-name regex:bwd_partial_dlogits_entropy \
		--launch-skip 133 --launch-count 1 \
		--set full --clock-control=none \
		-f -o $(ENTROPY_BWD_OUTPUT_NAME) \
		pytest -s -v tests/pytorch/test_linear_cross_entropy_with_entropy.py

entropy-ncu-bwd-cli:
	ONLY_PROFILE=1 ncu --kernel-name regex:bwd_partial_dlogits_entropy \
		--launch-skip 2 --launch-count 1 \
		--clock-control=none \
		--log-file bwd_partial_dlogits_entropy.log \
		pytest -s -v tests/pytorch/test_linear_cross_entropy_with_entropy.py