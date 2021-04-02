#!/usr/bin/env bash

# An awful amount of tmux!
# Spins up 15 runners

function spawnThing() {
  tmux split-window ". ./env.sh; $*"
  tmux select-layout tiled  
}

# spawnThing './runner.py --trajectory circle --run 0 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 10 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 20 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory circle --run 1 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 11 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 21 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory circle --run 2 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 12 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 22 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory circle --run 3 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 13 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 23 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory circle --run 4 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 14 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory circle --run 24 --users 5 --radius 5 --download --no-upload'

# spawnThing './runner.py --trajectory bowtie --run 0 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 10 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 20 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory bowtie --run 1 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 11 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 21 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory bowtie --run 2 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 12 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 22 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory bowtie --run 3 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 13 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 23 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory bowtie --run 4 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 14 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory bowtie --run 24 --users 5 --radius 5 --download --no-upload'

# spawnThing './runner.py --trajectory ladder --run 0 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 10 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 20 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory ladder --run 1 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 11 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 21 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory ladder --run 2 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 12 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 22 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory ladder --run 3 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 13 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 23 --users 5 --radius 5 --download --no-upload'
# spawnThing './runner.py --trajectory ladder --run 4 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 14 --users 5 --radius 5 --download --no-upload; ./runner.py --trajectory ladder --run 24 --users 5 --radius 5 --download --no-upload'

# for MODE in {nm,pso}; do
#   THR=4
  
#   spawnThing "./optimizerunner.py --run 0 --mode ${MODE} --threads ${THR} --energy -250; ./optimizerunner.py --run 1 --mode ${MODE} --threads ${THR} --energy -250; ; ./optimizerunner.py --run 2 --mode ${MODE} --threads ${THR} --energy -250"
#   spawnThing "./optimizerunner.py --run 0 --mode ${MODE} --threads ${THR} --energy -750; ./optimizerunner.py --run 1 --mode ${MODE} --threads ${THR} --energy -750; ; ./optimizerunner.py --run 2 --mode ${MODE} --threads ${THR} --energy -750"
  
#   spawnThing "./optimizerunner.py --run 1 --mode ${MODE} --threads ${THR} --energy -500; ./optimizerunner.py --run 2 --mode ${MODE} --threads ${THR} --energy -500"
#   spawnThing "./optimizerunner.py --run 1 --mode ${MODE} --threads ${THR} --energy -1000; ./optimizerunner.py --run 2 --mode ${MODE} --threads ${THR} --energy -1000"
# done

# 4 trajectories to judge, 32 cores, 8 on the inner loop 4 deep is 32 of 30 so we're goodish!
# Swap TRAJ between screens though, otherwise it'll be crowded...
for TRAJ in {nm,pso}-{0,500}Wh-r0; do
#for TRAJ in pso-{0,500}Wh-r0; do
  #for RUN in {0..4}; do
  for RUN in {5..9}; do
    CFG='--users 5 --radius 5 --download --no-upload'
    spawnThing "./runner2.py ${CFG} --trajectory ${TRAJ} --run ${RUN}; ./runner2.py ${CFG} --trajectory ${TRAJ} --run 1${RUN}; ./runner2.py ${CFG} --trajectory ${TRAJ} --run 2${RUN}"
  done
done

# Back to main
tmux select-pane -t 0

echo Started the stuff
