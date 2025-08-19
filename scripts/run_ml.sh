MODEL=$1
TAG=$2

python main_ml.py --model $MODEL --raw-csv ../database/external/Tg_merged.csv --labels Tg --tag $TAG --optimize-hparams
python main_ml.py --model $MODEL --raw-csv ../database/external/FFV_merged.csv --labels FFV --tag $TAG --optimize-hparams
python main_ml.py --model $MODEL --raw-csv ../database/external/Tc_merged.csv --labels Tc --tag $TAG --optimize-hparams
python main_ml.py --model $MODEL --raw-csv ../database/external/Rg_relative_difference_merged.csv --labels Rg --tag $TAG --optimize-hparams
python main_ml.py --model $MODEL --raw-csv ../database/internal/train.csv  --labels Density --tag $TAG --optimize-hparams
