# StruggleTab Benchmark



### Environment 
```bash
conda create --name sd python==3.12
conda activate sd
pip install pandas kagglehub[pandas-datasets] optuna
```

### Steps to run
Install git LFS first if you don't have it: <br>
https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab


```bash
chmod +x acquisition.sh
./acquisition.sh
python handler.py moma # check if handler works
```

How to use inside a model pipeline:
```python
sys.path.append(path_to_struggletab_codebase)
from handler import UnifiedDataLoader; from eval import evaluate
train_df = loader.get_train_data()

synthetic_data_frame = your_full_model_pipeline() # run your model training/sampling code

result = evaluate(dataset_name,  synthetic_data_frame)
```

### Attributions
- [Honeypot](secrepo.com/honeypot/honeypot.json.zip) from Mike Sconzo, SecRepo
- [Stroke](kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle / Federico Soriano Palacios
- [Museum of Modern Art (MoMA) Collection](https://github.com/MuseumofModernArt/collection/tree/e7dbe23cbe87022831190632fdca26b568f8d351), v2026-06-02
- [CERN](opendata.cern.ch/record/304) from Thomas McCauley, CERN
- [High Frequency Crypto Limit Order Book Data](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data) shared on Kaggle by Martin Søgaard Nielsen
- [Olist](kaggle.com/dsv/195341) from André Sionek, Olist
