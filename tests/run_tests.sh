#!/usr/bin/env bash
# Run each test in its own pytest process.
#
# Torch XPU / SYCL state left behind by one model test can deadlock the next when
# they share an interpreter, so tests are not run in a single process. Extra
# arguments select tests, e.g. ./tests/run_tests.sh tests/test_models_torch.py -k
# "not sparse".
set -uo pipefail

TIMEOUT=${TIMEOUT:-120}
cd "$(dirname "$0")/.."
targets=("${@:-tests}")

mapfile -t test_ids < <(python -m pytest --collect-only -q "${targets[@]}" |
    grep '::')
if [ ${#test_ids[@]} -eq 0 ]; then
    echo "No tests collected for: ${targets[*]}" >&2
    exit 1
fi

failed=()
for test_id in "${test_ids[@]}"; do
    timeout "$TIMEOUT" python -m pytest "$test_id"
    # Exit code 1 is a test failure, 124 a timeout, 134 an abort during shutdown.
    status=$?
    [ $status -eq 0 ] || failed+=("$test_id (exit $status)")
done

echo
echo "Ran ${#test_ids[@]} tests, ${#failed[@]} failed."
for test_id in "${failed[@]}"; do
    echo "FAILED $test_id"
done
[ ${#failed[@]} -eq 0 ]
