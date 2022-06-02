#!/bin/bash
> results.txt
touch results.txt
for img in 720p 1080p 4k
do
echo "----------------------------------------" >> results.txt
echo "Running $img test with 1, 4, 16 blocks"
for blocks in 1 4 16
do
echo "Running $img test with 1, 256, 1024 threads"
for threads	in 1 256 1024
do
echo "Running $img test with $blocks blocks and $threads threads"
echo -e "\n>> $img test with $blocks blocks and $threads threads" >> results.txt
./gray ./Images/${img}.png ./Output/${img}_"${blocks}"_x_"${threads}".png "${blocks}" "${threads}" >> results.txt
done
done
done
