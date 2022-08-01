#!/bin/bash
find ./figures/ -name "*.pdf" -type f -delete
find ./figures/ -name "*.png" -type f -delete
#find ./data/ -name "*.txt" -type f -delete
python "main.py"