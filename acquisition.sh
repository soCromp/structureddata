#!/bin/bash
# acquisition.sh

DATA_DIR="data/raw"
mkdir -p $DATA_DIR
cd $DATA_DIR

############ Honeypot ############
wget https://secrepo.com/honeypot/honeypot.json.zip
unzip -o honeypot.json
rm -f honeypot.json.zip*
echo "Honeypot saved locally."

############ MOMA ############
MOMA_DIR="./moma"
MOMA_HASH="e7dbe23cbe87022831190632fdca26b568f8d351"

mkdir -p $MOMA_DIR
cd $MOMA_DIR

# 1. Clone the repo (this pulls the LFS files natively)
git clone https://github.com/MuseumofModernArt/collection.git

# 2. Pin to your specific commit
cd collection
git checkout $MOMA_HASH

echo "MoMA saved locally."

