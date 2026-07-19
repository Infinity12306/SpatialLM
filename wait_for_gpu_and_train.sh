#!/bin/bash

# Wait for available GPUs and launch commands in sequence.
# For each command: find a free GPU, launch in background, wait 10 minutes,
# then check its log for startup CUDA OOM before dispatching the next command.

set -e

# Configuration
MEMORY_THRESHOLD_MB=1000  # Maximum memory usage (MB) to consider GPU "available"
POLL_INTERVAL_SEC=10      # Seconds between GPU checks
MAX_GPUS=8                # Total number of GPUs in the system
POST_LAUNCH_WAIT_SEC=600  # 10 minutes
OOM_KEYWORD="Error"

COMMANDS=(
    "bash /data2/chenjq24/SpatialLM/run_scripts/run_train_stage2_filtered_point_tokens.sh"
)

LOG_FILES=(
    ""
)

commands_from_args=false

# Parse optional command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --memory-threshold)
            MEMORY_THRESHOLD_MB="$2"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL_SEC="$2"
            shift 2
            ;;
        --max-gpus)
            MAX_GPUS="$2"
            shift 2
            ;;
        --)
            shift
            COMMANDS=("$@")
            commands_from_args=true
            break
            ;;
        *)
            # Treat remaining args as commands (each command should be quoted)
            COMMANDS=("$@")
            commands_from_args=true
            break
            ;;
    esac
done

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
    echo "No commands provided."
    echo "Usage: $0 [--memory-threshold MB] [--poll-interval SEC] [--max-gpus N] -- \"cmd1\" \"cmd2\" ..."
    exit 1
fi

if [[ "$commands_from_args" == true ]]; then
    LOG_FILES=()
    timestamp=$(date '+%Y%m%d_%H%M%S')
    for idx in "${!COMMANDS[@]}"; do
        LOG_FILES+=("logs/wait_for_gpu_and_train_${timestamp}_cmd_${idx}.log")
    done
fi

if [[ ${#COMMANDS[@]} -ne ${#LOG_FILES[@]} ]]; then
    echo "Mismatch: COMMANDS has ${#COMMANDS[@]} entries but LOG_FILES has ${#LOG_FILES[@]} entries."
    echo "Please provide one log file per command."
    exit 1
fi

echo "=========================================="
echo "GPU Polling Script"
echo "=========================================="
echo "Configuration:"
echo "  Memory threshold: ${MEMORY_THRESHOLD_MB} MB"
echo "  Poll interval: ${POLL_INTERVAL_SEC} seconds"
echo "  Max GPUs to check: ${MAX_GPUS}"
echo "  Post-launch wait: ${POST_LAUNCH_WAIT_SEC} seconds"
echo "  OOM keyword: ${OOM_KEYWORD}"
echo "  Number of commands: ${#COMMANDS[@]}"
echo "=========================================="
echo ""

# Function to check GPU memory usage
check_gpu_memory() {
    local gpu_id=$1
    # Get used memory in MB (using nvidia-smi)
    local used_mem_mb
    used_mem_mb=$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "$used_mem_mb"
}

next_cmd_idx=0

while (( next_cmd_idx < ${#COMMANDS[@]} )); do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GPU availability..."
    launched_this_round=false

    for gpu_id in $(seq 0 $((MAX_GPUS - 1))); do
        used_mem_mb=$(check_gpu_memory "$gpu_id")
        echo "  GPU $gpu_id: ${used_mem_mb} MB used"

        if [ "$used_mem_mb" -lt "$MEMORY_THRESHOLD_MB" ]; then
            cmd="${COMMANDS[$next_cmd_idx]}"
            log_file="${LOG_FILES[$next_cmd_idx]}"

            if [[ -n "$log_file" ]]; then
                launch_cmd="${cmd} > \"${log_file}\" 2>&1 &"
                mkdir -p "$(dirname "$log_file")"
            else
                launch_cmd="${cmd} &"
            fi

            echo ""
            echo "=========================================="
            echo "GPU ${gpu_id} is available! (using ${used_mem_mb} MB)"
            echo "Launching command index ${next_cmd_idx}:"
            echo "  ${cmd}"
            if [[ -n "$log_file" ]]; then
                echo "Log file:"
                echo "  ${log_file}"
            else
                echo "Log file:"
                echo "  <none; command stdout/stderr are not redirected by this script>"
            fi
            echo "=========================================="
            echo ""

            CUDA_VISIBLE_DEVICES="$gpu_id" nohup bash -c "$launch_cmd"
            launched_this_round=true

            if [[ -n "$log_file" ]]; then
                echo "Waiting ${POST_LAUNCH_WAIT_SEC} seconds before checking ${log_file}..."
            else
                echo "Waiting ${POST_LAUNCH_WAIT_SEC} seconds; no log file was configured for OOM checking..."
            fi
            sleep "$POST_LAUNCH_WAIT_SEC"

            if [[ -z "$log_file" ]]; then
                echo "No log file configured, skipping '${OOM_KEYWORD}' check."
                echo "Command index ${next_cmd_idx} is accepted; moving to the next command."
                next_cmd_idx=$((next_cmd_idx + 1))

                if (( next_cmd_idx >= ${#COMMANDS[@]} )); then
                    echo "All commands have been dispatched."
                    exit 0
                fi
            elif [[ -f "$log_file" ]] && grep -q "$OOM_KEYWORD" "$log_file"; then
                echo "Detected '${OOM_KEYWORD}' in ${log_file}."
                echo "Command index ${next_cmd_idx} likely launched on an already occupied GPU."
                echo "Will retry the same command after polling for free GPUs again."
            else
                echo "No '${OOM_KEYWORD}' found in ${log_file}."
                echo "Command index ${next_cmd_idx} is accepted; moving to the next command."
                next_cmd_idx=$((next_cmd_idx + 1))

                if (( next_cmd_idx >= ${#COMMANDS[@]} )); then
                    echo "All commands have been dispatched."
                    exit 0
                fi
            fi

            break
        fi
    done

    if [[ "$launched_this_round" == false ]]; then
        echo "  No available GPUs found for remaining commands. Waiting ${POLL_INTERVAL_SEC} seconds..."
        echo ""
        sleep "$POLL_INTERVAL_SEC"
    fi
done
