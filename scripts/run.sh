MODEL=$1
TAG=$2

python main_dl.py --model $MODEL --raw-csv-path database/external/Tg_merged.csv --labels Tg --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250
python main_dl.py --model $MODEL --raw-csv-path database/external/FFV_merged.csv --labels FFV --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250
python main_dl.py --num-epochs 1000 --model $MODEL --raw-csv-path database/external/Tc_merged.csv --labels Tc --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250
python main_dl.py --num-epochs 1000 --model $MODEL --raw-csv-path database/external/Rg_relative_difference_merged.csv --labels Rg --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250
python main_dl.py --num-epochs 1000 --model $MODEL --labels Density --run-production --optimize-hparams --tag $TAG --early-stopping-patience 250
