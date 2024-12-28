CONFIG="config/test.json"

python src/preprocess/MS.py --config $CONFIG
python src/learning.py --config $CONFIG
python src/analysis.py --config $CONFIG
python src/comprehension.py --config $CONFIG