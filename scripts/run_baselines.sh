#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 SEED" >&2
    exit 2
fi
SEED="$1"
cd "$(dirname "$0")/.."
python eval_baselines.py --env-list data/scenarios/map/train_envs.json --output-dir "results/main/map/rules/seed_${SEED}/train" --seed "$SEED" --policies static_cio random clairvoyant_greedy --greedy-mode current --num-candidates 64
python eval_baselines.py --env-list data/scenarios/map/test_envs.json --output-dir "results/main/map/rules/seed_${SEED}/test" --seed "$SEED" --policies static_cio random clairvoyant_greedy --greedy-mode current --num-candidates 64
python eval_baselines.py --env-list data/scenarios/map/train_envs.json --output-dir "results/main/map/rules/seed_${SEED}/train" --seed "$SEED" --policies clairvoyant_greedy --greedy-mode next --num-candidates 64
python eval_baselines.py --env-list data/scenarios/map/test_envs.json --output-dir "results/main/map/rules/seed_${SEED}/test" --seed "$SEED" --policies clairvoyant_greedy --greedy-mode next --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_a/train_envs.json --output-dir "results/main/site_a/rules/seed_${SEED}/train" --seed "$SEED" --policies static_cio random clairvoyant_greedy --greedy-mode current --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_a/test_envs.json --output-dir "results/main/site_a/rules/seed_${SEED}/test" --seed "$SEED" --policies static_cio random clairvoyant_greedy --greedy-mode current --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_a/train_envs.json --output-dir "results/main/site_a/rules/seed_${SEED}/train" --seed "$SEED" --policies clairvoyant_greedy --greedy-mode next --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_a/test_envs.json --output-dir "results/main/site_a/rules/seed_${SEED}/test" --seed "$SEED" --policies clairvoyant_greedy --greedy-mode next --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_b/train_envs.json --output-dir "results/main/site_b/rules/seed_${SEED}/train" --seed "$SEED" --policies static_cio random clairvoyant_greedy --greedy-mode current --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_b/test_envs.json --output-dir "results/main/site_b/rules/seed_${SEED}/test" --seed "$SEED" --policies static_cio random clairvoyant_greedy --greedy-mode current --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_b/train_envs.json --output-dir "results/main/site_b/rules/seed_${SEED}/train" --seed "$SEED" --policies clairvoyant_greedy --greedy-mode next --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_b/test_envs.json --output-dir "results/main/site_b/rules/seed_${SEED}/test" --seed "$SEED" --policies clairvoyant_greedy --greedy-mode next --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_c/train_envs.json --output-dir "results/main/site_c/rules/seed_${SEED}/train" --seed "$SEED" --policies static_cio random clairvoyant_greedy --greedy-mode current --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_c/test_envs.json --output-dir "results/main/site_c/rules/seed_${SEED}/test" --seed "$SEED" --policies static_cio random clairvoyant_greedy --greedy-mode current --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_c/train_envs.json --output-dir "results/main/site_c/rules/seed_${SEED}/train" --seed "$SEED" --policies clairvoyant_greedy --greedy-mode next --num-candidates 64
python eval_baselines.py --env-list data/scenarios/site_c/test_envs.json --output-dir "results/main/site_c/rules/seed_${SEED}/test" --seed "$SEED" --policies clairvoyant_greedy --greedy-mode next --num-candidates 64
