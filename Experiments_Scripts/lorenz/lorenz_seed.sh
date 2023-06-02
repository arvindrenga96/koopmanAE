#!/bin/bash

# Name of your existing script
script="lorenz.sh"

# Loop 10 times
for i in {1..10}
do
  # Generate a random seed
  seed=$RANDOM

  # Call your existing script with the new seed and run in parallel
  bash "$script" "$1/$seed" "$seed" &
done