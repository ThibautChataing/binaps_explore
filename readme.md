# BINAPS
binary autoencoder interpretable

*Coming from : https://eda.mmci.uni-saarland.de/pubs/2021/binaps-fischer,vreeken.pdf*
Code source : http://eda.mmci.uni-saarland.de/prj/binaps/


## Input data
### Format
Data expected : transaction file (sparse binary matrix representation). For better use of memory space, files represent each matrix row on a separate line, where each non-zero entry is given by the corresponding index separated by whitespace.
For example, the matrix

	1	0	0	0
	0	1	1	0
	1	0	0	1

is given by the following lines in a .dat file:

    1
    2 3
    1 4

Files name are my_file.dat  

### Synthetic
Need R installed 
```
$sudo apt update
$sudo apt -y upgrade
$sudo apt install apt-transport-https software-properties-common
$sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57\\CBB651716619E084DAB9
$sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
$sudo apt update
$sudo apt install r-base
```
Running the program give for each experiment 4 files
- file.dat
- file.dat_patterns.txt
- file_itemOverlap.dat
- file_itemOverlap.dat_patterns.txt
Data generated will be under the format : "and_synthetic_scale_${number of patterns}_${number of rows}_${max pattern size}_${% of noise}_${density}"
"Generating data of $n rows and $m different patterns with $noise% noise, $density marginal density and patterns of size 2 to $maxPatSize.\n"

## Training
### Input
file described as in synthetic
### Output
input_file.binaps.patterns
### Easy run
```
$python Binaps_code/main.py -i Data/synth/data/and_synthetic_scaleSamples_100_5000_10_0.001_0.05.dat
```
### F1 or Jaccard
binaps_explore/Data/Synthetic_data/comp_patterns.py
```
$python Data/Synthetic_data/comp_patterns.py -p Data/synth/data/and_synthetic_scaleSamples_100_5000_10_0.001_0.05.binaps.patterns -t Binaps -r Data/synth/data/and_synthetic_scaleSamples_100_5000_10_0.001_0.05.dat_patterns.txt -m F1
``