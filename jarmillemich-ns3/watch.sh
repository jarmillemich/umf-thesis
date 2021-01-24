#!/usr/bin/env bash

# Initial build
clear
./waf

# Watch and build again
while true; do
	inotifywait -r -e modify scratch-thesis
	sleep 2
	clear
	echo === Building ===
	echo
	./waf $*
	echo
	echo === Built ===
done
