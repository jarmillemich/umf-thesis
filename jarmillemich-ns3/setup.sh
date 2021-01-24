#!/usr/bin/env bash

NS3=../ns-3-dev

# Create links to our mobility model
ln -rfs mobility-model/* $NS3/src/mobility/model/

# Create links to our scratch folder
ln -rfs scratch-thesis $NS3/scratch/thesis
