#!/bin/bash
# acquisition.sh


############ MOMA ############
MOMA_DIR="/home/sonia/structureddata/data/raw/moma"
MOMA_HASH="e7dbe23cbe87022831190632fdca26b568f8d351"

mkdir -p $MOMA_DIR
cd $MOMA_DIR

# 1. Clone the repo (this pulls the LFS files natively)
git clone https://github.com/MuseumofModernArt/collection.git

# 2. Pin to your specific commit
cd collection
git checkout $MOMA_HASH

echo "MoMA dataset pinned and saved locally."

