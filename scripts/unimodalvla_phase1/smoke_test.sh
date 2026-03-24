set -x

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python test/test_unimodalvla_phase1.py
