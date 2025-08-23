MODEL=$1
TAG=$2
N_TRIALS=${3:-25}

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $SCRIPT_DIR

python "$SCRIPT_DIR/main_dl.py" --model $MODEL --raw-csv "$SCRIPT_DIR/../database/external/Tg_merged.csv" --labels Tg --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250 --n-trials $N_TRIALS
python "$SCRIPT_DIR/main_dl.py" --model $MODEL --raw-csv "$SCRIPT_DIR/../database/external/FFV_merged.csv" --labels FFV --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250 --n-trials $N_TRIALS
python "$SCRIPT_DIR/main_dl.py" --num-epochs 1000 --model $MODEL --raw-csv "$SCRIPT_DIR/../database/external/Tc_merged.csv" --labels Tc --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250 --n-trials $N_TRIALS
python "$SCRIPT_DIR/main_dl.py" --num-epochs 1000 --model $MODEL --raw-csv "$SCRIPT_DIR/../database/external/Rg_relative_difference_merged.csv" --labels Rg --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250 --n-trials $N_TRIALS
python "$SCRIPT_DIR/main_dl.py" --num-epochs 1000 --model $MODEL --raw-csv "$SCRIPT_DIR/../database/internal/train.csv"  --labels Density --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250 --n-trials $N_TRIALS
