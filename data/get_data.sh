#!/bin/bash

if [ ! -d "ig02-cars" ]; then
  echo "[*] Downloading dataset"
  wget https://lear.inrialpes.fr/people/marszalek/data/ig02/ig02-v1.0-cars.zip
  echo "[*] Unzipping file"
  unzip ig02-v1.0-cars.zip -d ig02-cars
  rm ig02-v1.0-cars.zip
fi

echo "[*] Creating training and evaluation file lists"
python process_txt.py

echo "[*] Generating custom labels"
python gen_labels.py

