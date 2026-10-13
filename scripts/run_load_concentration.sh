#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 SEED" >&2
    exit 2
fi
SEED="$1"
cd "$(dirname "$0")/.."
python eval_load_concentration.py --run-dir "results/main/map/sac_h1/seed_${SEED}" --env-list data/scenarios/map/test_envs.json --output "results/load_concentration/map/sac_h1/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/map/sac_h5/seed_${SEED}" --env-list data/scenarios/map/test_envs.json --output "results/load_concentration/map/sac_h5/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/map/link_mvd_h5/seed_${SEED}" --env-list data/scenarios/map/test_envs.json --output "results/load_concentration/map/link_mvd_h5/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/site_a/sac_h1/seed_${SEED}" --env-list data/scenarios/site_a/test_envs.json --output "results/load_concentration/site_a/sac_h1/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/site_a/sac_h5/seed_${SEED}" --env-list data/scenarios/site_a/test_envs.json --output "results/load_concentration/site_a/sac_h5/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/site_a/link_mvd_h5/seed_${SEED}" --env-list data/scenarios/site_a/test_envs.json --output "results/load_concentration/site_a/link_mvd_h5/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/site_b/sac_h1/seed_${SEED}" --env-list data/scenarios/site_b/test_envs.json --output "results/load_concentration/site_b/sac_h1/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/site_b/sac_h5/seed_${SEED}" --env-list data/scenarios/site_b/test_envs.json --output "results/load_concentration/site_b/sac_h5/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/site_b/link_mvd_h5/seed_${SEED}" --env-list data/scenarios/site_b/test_envs.json --output "results/load_concentration/site_b/link_mvd_h5/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/site_c/sac_h1/seed_${SEED}" --env-list data/scenarios/site_c/test_envs.json --output "results/load_concentration/site_c/sac_h1/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/site_c/sac_h5/seed_${SEED}" --env-list data/scenarios/site_c/test_envs.json --output "results/load_concentration/site_c/sac_h5/seed_${SEED}/evaluation.json" --seed "$SEED"
python eval_load_concentration.py --run-dir "results/main/site_c/link_mvd_h5/seed_${SEED}" --env-list data/scenarios/site_c/test_envs.json --output "results/load_concentration/site_c/link_mvd_h5/seed_${SEED}/evaluation.json" --seed "$SEED"
