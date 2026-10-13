#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 SEED" >&2
    exit 2
fi
SEED="$1"
cd "$(dirname "$0")/.."
python train_sac.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/main/map/sac_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 10
python eval_sac.py --run-dir "results/main/map/sac_h1/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/main/map/sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 10
python eval_sac.py --run-dir "results/main/map/sac_h5/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_transformer_sac.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/main/map/transformer_sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 10
python eval_transformer_sac.py --run-dir "results/main/map/transformer_sac_h5/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_cql.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/main/map/cql_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 10
python eval_cql.py --run-dir "results/main/map/cql_h1/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_iql.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/main/map/iql_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 10
python eval_iql.py --run-dir "results/main/map/iql_h1/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/map/train_manifest.json --output-dir "results/main/map/link_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 10
python eval_link_mvd.py --run-dir "results/main/map/link_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/map/train_envs.json --test-env-list data/scenarios/map/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/main/site_a/sac_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/main/site_a/sac_h1/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/main/site_a/sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/main/site_a/sac_h5/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_transformer_sac.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/main/site_a/transformer_sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_transformer_sac.py --run-dir "results/main/site_a/transformer_sac_h5/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_cql.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/main/site_a/cql_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_cql.py --run-dir "results/main/site_a/cql_h1/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_iql.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/main/site_a/iql_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_iql.py --run-dir "results/main/site_a/iql_h1/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_a/train_manifest.json --output-dir "results/main/site_a/link_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/main/site_a/link_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/site_a/train_envs.json --test-env-list data/scenarios/site_a/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/main/site_b/sac_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/main/site_b/sac_h1/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/main/site_b/sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/main/site_b/sac_h5/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_transformer_sac.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/main/site_b/transformer_sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_transformer_sac.py --run-dir "results/main/site_b/transformer_sac_h5/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_cql.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/main/site_b/cql_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_cql.py --run-dir "results/main/site_b/cql_h1/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_iql.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/main/site_b/iql_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_iql.py --run-dir "results/main/site_b/iql_h1/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_b/train_manifest.json --output-dir "results/main/site_b/link_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/main/site_b/link_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/site_b/train_envs.json --test-env-list data/scenarios/site_b/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/main/site_c/sac_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/main/site_c/sac_h1/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_sac.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/main/site_c/sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_sac.py --run-dir "results/main/site_c/sac_h5/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_transformer_sac.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/main/site_c/transformer_sac_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_transformer_sac.py --run-dir "results/main/site_c/transformer_sac_h5/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_cql.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/main/site_c/cql_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_cql.py --run-dir "results/main/site_c/cql_h1/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_iql.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/main/site_c/iql_h1/seed_${SEED}" --history-length 1 --seed "$SEED" --reward-scale 1
python eval_iql.py --run-dir "results/main/site_c/iql_h1/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
python train_link_mvd.py --dataset offline_data/rollouts_64/site_c/train_manifest.json --output-dir "results/main/site_c/link_mvd_h5/seed_${SEED}" --history-length 5 --seed "$SEED" --reward-scale 1
python eval_link_mvd.py --run-dir "results/main/site_c/link_mvd_h5/seed_${SEED}" --train-env-list data/scenarios/site_c/train_envs.json --test-env-list data/scenarios/site_c/test_envs.json --seed "$SEED"
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
