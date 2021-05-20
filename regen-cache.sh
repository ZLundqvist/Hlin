python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd none --window-size 11 --window-step-size 6 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd none --window-size 23 --window-step-size 12 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd none --window-size 7 --window-step-size 4 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd none --window-size 10 --window-step-size 5 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd none --window-size 11 --window-step-size 11 --skip-training

python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd label --window-size 11 --window-step-size 6 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd label --window-size 23 --window-step-size 12 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd label --window-size 7 --window-step-size 4 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd label --window-size 10 --window-step-size 5 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd label --window-size 11 --window-step-size 11 --skip-training

python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd first --window-size 11 --window-step-size 6 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd first --window-size 23 --window-step-size 12 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd first --window-size 7 --window-step-size 4 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd first --window-size 10 --window-step-size 5 --skip-training
python src/core.py -m isolation_forest -p sliding_window --input-dir input/ --dd first --window-size 11 --window-step-size 11 --skip-training




python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd none --delta-t 25 --skip-training
python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd none --delta-t 50 --skip-training
python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd none --delta-t 100 --skip-training
python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd none --delta-t 200 --skip-training

python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd label --delta-t 25 --skip-training
python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd label --delta-t 50 --skip-training
python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd label --delta-t 100 --skip-training
python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd label --delta-t 200 --skip-training

python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd first --delta-t 25 --skip-training
python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd first --delta-t 50 --skip-training
python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd first --delta-t 100 --skip-training
python src/core.py -m isolation_forest -p frequency_vector --input-dir input/ --dd first --delta-t 200 --skip-training




python src/core.py -m isolation_forest -p n_gram --input-dir input/ --dd none --ngram-size 3 --skip-training
python src/core.py -m isolation_forest -p n_gram --input-dir input/ --dd none --ngram-size 5 --skip-training
python src/core.py -m isolation_forest -p n_gram --input-dir input/ --dd none --ngram-size 10 --skip-training

python src/core.py -m isolation_forest -p n_gram --input-dir input/ --dd label --ngram-size 3 --skip-training
python src/core.py -m isolation_forest -p n_gram --input-dir input/ --dd label --ngram-size 5 --skip-training
python src/core.py -m isolation_forest -p n_gram --input-dir input/ --dd label --ngram-size 10 --skip-training

python src/core.py -m isolation_forest -p n_gram --input-dir input/ --dd first --ngram-size 3 --skip-training
python src/core.py -m isolation_forest -p n_gram --input-dir input/ --dd first --ngram-size 5 --skip-training
python src/core.py -m isolation_forest -p n_gram --input-dir input/ --dd first --ngram-size 10 --skip-training