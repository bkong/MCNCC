#!/bin/bash

DATASET=$1
TARGET_DIR="./datasets"
case $DATASET in
    "fid300")
        URL="https://fid.dmi.unibas.ch/FID-300.zip"
        ZIP_FILE="FID-300.zip"
        wget -N $URL -O $ZIP_FILE
        unzip -d $TARGET_DIR $ZIP_FILE
        rm $ZIP_FILE
        ;;
    "israeli")
        echo "Contact the authors of https://www.researchgate.net/publication/280803567 for the dataset."
        ;;
    "maps")
        URL="https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/maps.zip"
        ZIP_FILE="maps.zip"
        wget -N $URL -O $ZIP_FILE
        unzip -d $TARGET_DIR $ZIP_FILE
        rm $ZIP_FILE
        ;;
    "facades")
        URL="http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip"
        ZIP_FILE="CMP_facade_DB_base.zip"
        wget -N $URL -O $ZIP_FILE
        unzip -d $TARGET_DIR $ZIP_FILE
        rm $ZIP_FILE

        URL="http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip"
        ZIP_FILE="CMP_facade_DB_extended.zip"
        wget -N $URL -O $ZIP_FILE
        unzip -d $TARGET_DIR $ZIP_FILE
        rm $ZIP_FILE

        TARGET_DIR="${TARGET_DIR}/facades"
        mkdir "${TARGET_DIR}/facades"
        mv ${TARGET_DIR}/base/*.jpg ${TARGET_DIR}/facades/
        mv ${TARGET_DIR}/base/*.png ${TARGET_DIR}/facades/
        mv ${TARGET_DIR}/extended/*.jpg ${TARGET_DIR}/facades/
        mv ${TARGET_DIR}/extended/*.png ${TARGET_DIR}/facades/
        mv ${TARGET_DIR}/label_names.txt ${TARGET_DIR}/readme.txt ${TARGET_DIR}/facades/
        rm -rf ${TARGET_DIR}/base ${TARGET_DIR}/extended
        ;;
    *)
        echo "Unknown dataset!"
        ;;
esac
