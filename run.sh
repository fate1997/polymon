python main_dl.py --model gatv2 --raw-csv-path database/external/Tg_merged.csv --labels Tg --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 250
python main_dl.py --model gatv2 --raw-csv-path database/external/FFV_merged.csv --labels FFV --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 250
python main_dl.py --num-epochs 1000 --model gatv2 --raw-csv-path database/external/Tc_merged.csv --labels Tc --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 250
python main_dl.py --num-epochs 1000 --model gatv2 --raw-csv-path database/external/Rg_relative_difference_merged.csv --labels Rg --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 250
python main_dl.py --num-epochs 1000 --model gatv2 --labels Density --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 250

python main_dl.py --model gatport --raw-csv-path database/external/Tg_merged.csv --labels Tg --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 250
python main_dl.py --model gatport --raw-csv-path database/external/FFV_merged.csv --labels FFV --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 250
python main_dl.py --num-epochs 1000 --model gatport --raw-csv-path database/external/Tc_merged.csv --labels Tc --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 250
python main_dl.py --num-epochs 1000 --model gatport --labels Density --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 250
python main_dl.py --num-epochs 1000 --model gatport --raw-csv-path database/external/Rg_relative_difference_merged.csv --labels Rg --run-production --optimize-hparams --tag hparams-opt --early-stopping-patience 500
