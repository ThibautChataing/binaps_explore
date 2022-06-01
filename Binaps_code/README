
This is a Python implementation of BinaPs based on PyTorch licensed under the GNU GENERAL PUBLIC LICENSE version 3.0.

We generally recommend to use a GPU to train BinaPs, but the code is generally applicable also on CPUs.
When only a CPU is available, the code will automatically run purely on CPU.

Required python packages for BinaPs are:
	Pytorch
	Scipy
	Pandas
	Numpy

To access available arguments, use

	python main.py -h



The code expects a transaction file as input (i.e. sparse binary matrix representation).
These files -- commonly suffixed with ".dat" -- represent each matrix row on a separate line, where each non-zero entry is given by the corresponding index separated by whitespace.
For example, the matrix

	1	0	0	0
	0	1	1	0
	1	0	0	1

is given by the following lines in a .dat file:

1
2 3
1 4



An example call of BinaPs with default parameters, changing the batch size:

python main.py --input input_filename.dat --batch_size 32


This call will generate a file called input_filename.binaps.patterns containing a set of (nonempty) patterns, where each pattern is given on a separate line.
