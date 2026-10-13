#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 SEED" >&2
    exit 2
fi
SEED="$1"
cd "$(dirname "$0")/.."
python train_sac.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/sac_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 10
python eval_sac.py --run-dir "results/history/map/sac_h1/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/sac_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 10
python eval_sac.py --run-dir "results/history/map/sac_h3/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 10
python eval_sac.py --run-dir "results/history/map/sac_h5/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/sac_mvd_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 10
python eval_sac_mvd.py --run-dir "results/history/map/sac_mvd_h1/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/sac_mvd_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 10
python eval_sac_mvd.py --run-dir "results/history/map/sac_mvd_h3/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/sac_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 10
python eval_sac_mvd.py --run-dir "results/history/map/sac_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/link_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 10
python eval_link.py --run-dir "results/history/map/link_h1/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/link_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 10
python eval_link.py --run-dir "results/history/map/link_h3/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/link_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 10
python eval_link.py --run-dir "results/history/map/link_h5/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/link_mvd_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 10
python eval_link_mvd.py --run-dir "results/history/map/link_mvd_h1/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/link_mvd_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 10
python eval_link_mvd.py --run-dir "results/history/map/link_mvd_h3/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/history/map/link_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 10
python eval_link_mvd.py --run-dir "results/history/map/link_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/sac_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/history/site_a/sac_h1/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/sac_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/history/site_a/sac_h3/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/history/site_a/sac_h5/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/sac_mvd_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_sac_mvd.py --run-dir "results/history/site_a/sac_mvd_h1/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/sac_mvd_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_sac_mvd.py --run-dir "results/history/site_a/sac_mvd_h3/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/sac_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_sac_mvd.py --run-dir "results/history/site_a/sac_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/link_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_link.py --run-dir "results/history/site_a/link_h1/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/link_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_link.py --run-dir "results/history/site_a/link_h3/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/link_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_link.py --run-dir "results/history/site_a/link_h5/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/link_mvd_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/history/site_a/link_mvd_h1/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/link_mvd_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/history/site_a/link_mvd_h3/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/history/site_a/link_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/history/site_a/link_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/sac_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/history/site_b/sac_h1/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/sac_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/history/site_b/sac_h3/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/history/site_b/sac_h5/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/sac_mvd_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_sac_mvd.py --run-dir "results/history/site_b/sac_mvd_h1/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/sac_mvd_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_sac_mvd.py --run-dir "results/history/site_b/sac_mvd_h3/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/sac_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_sac_mvd.py --run-dir "results/history/site_b/sac_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/link_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_link.py --run-dir "results/history/site_b/link_h1/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/link_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_link.py --run-dir "results/history/site_b/link_h3/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/link_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_link.py --run-dir "results/history/site_b/link_h5/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/link_mvd_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/history/site_b/link_mvd_h1/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/link_mvd_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/history/site_b/link_mvd_h3/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/history/site_b/link_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/history/site_b/link_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/sac_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/history/site_c/sac_h1/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/sac_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/history/site_c/sac_h3/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/history/site_c/sac_h5/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/sac_mvd_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_sac_mvd.py --run-dir "results/history/site_c/sac_mvd_h1/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/sac_mvd_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_sac_mvd.py --run-dir "results/history/site_c/sac_mvd_h3/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_sac_mvd.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/sac_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_sac_mvd.py --run-dir "results/history/site_c/sac_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/link_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_link.py --run-dir "results/history/site_c/link_h1/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/link_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_link.py --run-dir "results/history/site_c/link_h3/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_link.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/link_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_link.py --run-dir "results/history/site_c/link_h5/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/link_mvd_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/history/site_c/link_mvd_h1/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/link_mvd_h3/seed_${SEED}" --history-length 3 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/history/site_c/link_mvd_h3/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/history/site_c/link_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/history/site_c/link_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
