#!/bin/bash

# Configuration Arrays
ITER_OPTS=("1" "20" "200")
GPUS=("1" "2")
COMPILE_OPTS=("1" "0")
PRELOAD_OPTS=("1" "0")
PARAMS_PATH="params/examples/tBL_WSe2.yaml"

LOG_DIR="./output/test_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
START_TIME_GLOBAL=$SECONDS

echo "Starting PtyRAD Integration Test Matrix..."

for iter in "${ITER_OPTS[@]}"; do
    echo "================================================"
    echo " PHASE: $iter ITERATIONS "
    echo "================================================"
    
    for gpu in "${GPUS[@]}"; do
        for comp in "${COMPILE_OPTS[@]}"; do
            for pre in "${PRELOAD_OPTS[@]}"; do
                
                TEST_NAME="iter${iter}_gpu${gpu}_compile${comp}_preload${pre}"
                echo -n "Running $TEST_NAME... "

                # Start individual timer
                START_TEST=$SECONDS

                # Launcher logic
                if [ "$gpu" == "2" ]; then
                    # Using accelerate for multi-GPU
                    accelerate launch --multi_gpu --num_processes=2 ./tools/run_ptyrad.py \
                        --params_path "$PARAMS_PATH"\
                        --gpuid "acc" \
                        --n_iter "$iter" \
                        --compile "$comp" \
                        --preload "$pre" \
                        --output_path "$LOG_DIR/$TEST_NAME" \
                        > "$LOG_DIR/${TEST_NAME}.txt" 2>&1

                else
                    # Run command and capture exit code
                    python ./tools/run_ptyrad.py \
                        --params_path "$PARAMS_PATH"\
                        --gpuid 0 \
                        --n_iter "$iter" \
                        --compile "$comp" \
                        --preload "$pre" \
                        --output_path "$LOG_DIR/$TEST_NAME" \
                        > "$LOG_DIR/${TEST_NAME}.txt" 2>&1

                fi

                # End individual timer
                EXIT_CODE=$?
                END_TEST=$SECONDS
                DURATION=$((END_TEST - START_TEST))

                if [ $EXIT_CODE -eq 0 ]; then
                    echo "PASS ✅ (${DURATION}s)"
                else
                    echo "FAIL ❌ (${DURATION}s) - Check $LOG_DIR/${TEST_NAME}.txt"
                fi
                
            done
        done
    done
done

TOTAL_DURATION=$(($SECONDS - START_TIME_GLOBAL))
echo "================================================"
echo "All tests complete in $(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s."