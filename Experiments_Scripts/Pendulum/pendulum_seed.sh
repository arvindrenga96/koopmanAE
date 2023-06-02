#!/bin/bash

# Name of your existing script
script="pendulum.sh"

# Loop 10 times
for i in {1..5}
do
  # Generate a random seed
  seed=$RANDOM

  # Call your existing script with the new seed and run in parallel
  #bash "$script" "$1/$seed" "$seed" &

  bash "$script" "2.4_no_noise/$seed" "$seed" "0.0" &
  bash "$script" "2.4_with_noise/$seed" "$seed" "0.05" &
  bash "$script" "2.4_with_more_noise/$seed" "$seed" "0.15" &

done


