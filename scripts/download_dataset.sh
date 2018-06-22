#!/bin/bash

DATASET=$1
case $DATASET in
    "fid300")
        URL="https://fid.dmi.unibas.ch/FID-300.zip"
        TARGET_DIR="./datasets"
        ZIP_FILE="FID-300.zip"

        wget -N $URL -O $ZIP_FILE
        unzip -d $TARGET_DIR $ZIP_FILE
        rm $ZIP_FILE
        ;;
    "israeli")
        echo "Contact the authors of https://www.researchgate.net/publication/280803567 for the dataset."
        ;;
    *)
        echo "Unknown dataset!"
        ;;
esac
