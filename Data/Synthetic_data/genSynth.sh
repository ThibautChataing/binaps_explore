#! /bin/bash --

echo "\n\n Generating data sets for feature scale test...\n"
# Generate scale data sets
for m in {10..90..20}
do
    n=10000
    noise=0.001
    density=0.05
    maxPatSize=10
    filename="and_synthetic_scale_${m}_${n}_${maxPatSize}_${noise}_${density}"
    echo "Generating data of $n rows and $m different patterns with $noise% noise, $density marginal density and patterns of size 2 to $maxPatSize.\n"
    Rscript generate_toy.R AND $m $n $maxPatSize $filename $noise $density
    # python ../datToHDF5.py --input ${filename}.dat
    # python ../datToHDF5.py --input ${filename}_itemOverlap.dat
done
for m in {100..900..200} {1000..5000..2000}
do
    n=100000
    noise=0.001
    density=0.05
    maxPatSize=10
    filename="and_synthetic_scale_${m}_${n}_${maxPatSize}_${noise}_${density}"
    echo "Generating data of $n rows and $m different patterns with $noise% noise, $density marginal density and patterns of size 2 to $maxPatSize.\n"
    Rscript generate_toy.R AND $m $n $maxPatSize $filename $noise $density
done

echo "\n\n Generating data sets for noise robustness test...\n"
# Generate noise data sets
for noise in 0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05
do
    n=100000
    m=100
    density=0.05
    maxPatSize=10
    filename="and_synthetic_noise_${m}_${n}_${maxPatSize}_${noise}_${density}"
    echo "Generating data of $n rows and $m different patterns with $noise% noise, $density marginal density and patterns of size 2 to $maxPatSize.\n"
    Rscript generate_toy.R AND $m $n $maxPatSize $filename $noise $density
    # python ../datToHDF5.py --input ${filename}.dat
    # python ../datToHDF5.py --input ${filename}_itemOverlap.dat
done

echo "\n\n Generating data sets for sample scale test...\n"
# Generate scale data sets
for n in {100..10000..100}
do
    m=100
    noise=0.001
    density=0.05
    maxPatSize=10
    filename="and_synthetic_scaleSamples_${m}_${n}_${maxPatSize}_${noise}_${density}"
    echo "Generating data of $n rows and $m different patterns with $noise% noise, $density marginal density and patterns of size 2 to $maxPatSize.\n"
    Rscript generate_toy.R AND $m $n $maxPatSize $filename $noise $density
done