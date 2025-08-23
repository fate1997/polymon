MODEL=$1
TAG=$2

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $SCRIPT_DIR

python "$SCRIPT_DIR/main_ml.py" --model $MODEL --raw-csv "$SCRIPT_DIR/../database/external/Tg_merged.csv" --labels Tg --tag $TAG --optimize-hparams
python "$SCRIPT_DIR/main_ml.py" --model $MODEL --raw-csv "$SCRIPT_DIR/../database/external/FFV_merged.csv" --labels FFV --tag $TAG --optimize-hparams
python "$SCRIPT_DIR/main_ml.py" --model $MODEL --raw-csv "$SCRIPT_DIR/../database/external/Tc_merged.csv" --labels Tc --tag $TAG --optimize-hparams
python "$SCRIPT_DIR/main_ml.py" --model $MODEL --raw-csv "$SCRIPT_DIR/../database/external/Rg_relative_difference_merged.csv" --labels Rg --tag $TAG --optimize-hparams
python "$SCRIPT_DIR/main_ml.py" --model $MODEL --raw-csv "$SCRIPT_DIR/../database/internal/train.csv"  --labels Density --tag $TAG --optimize-hparams
