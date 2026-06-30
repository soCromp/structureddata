# Structured Data Challenge Benchmark



### Environment 
```bash
conda create --name sd python==3.12
conda activate sd
pip install pandas kagglehub[pandas-datasets]
```

### Steps to run
Install git LFS first if you don't have it: <br>
https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab


```bash
chmod +x acquisition.sh
./acquisition.sh
python handler.py moma
```

### Attributions
- [Museum of Modern Art (MoMA) Collection](https://github.com/MuseumofModernArt/collection/tree/e7dbe23cbe87022831190632fdca26b568f8d351), v2026-06-02, shared under CC0 license
- [High Frequency Crypto Limit Order Book Data](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data) shared on Kaggle by Martin Søgaard Nielsen, Version 1, shared under CC0 license
