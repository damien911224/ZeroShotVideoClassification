#!/usr/bin/env bash
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz -O ./GoogleNews-vectors-negative300.bin.gz
gunzip -c ./GoogleNews-vectors-negative300.bin.gz > ./GoogleNews-vectors-negative300.bin