#!/usr/bin/env bash

# An awful amount of tmux!
# Spins up 15 runners

function spawnThing() {
  tmux split-window ". ./env.sh; $*"
  tmux select-layout tiled  
}

spawnThing './runner.py --trajectory circle --run 0 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 10 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 20 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory circle --run 1 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 11 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 21 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory circle --run 2 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 12 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 22 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory circle --run 3 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 13 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 23 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory circle --run 4 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 14 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 24 --users 5 --radius 5 --download --no-upload'

spawnThing './runner.py --trajectory bowtie --run 0 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 10 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 20 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory bowtie --run 1 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 11 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 21 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory bowtie --run 2 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 12 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 22 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory bowtie --run 3 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 13 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 23 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory bowtie --run 4 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 14 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 24 --users 5 --radius 5 --download --no-upload'

spawnThing './runner.py --trajectory ladder --run 0 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 10 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 20 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory ladder --run 1 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 11 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 21 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory ladder --run 2 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 12 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 22 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory ladder --run 3 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 13 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 23 --users 5 --radius 5 --download --no-upload'
spawnThing './runner.py --trajectory ladder --run 4 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 14 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 24 --users 5 --radius 5 --download --no-upload'

# Back to main
tmux select-pane -t 0

echo Started the stuff