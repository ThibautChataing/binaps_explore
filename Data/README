
This directory contains scripts to reconstruct the synthetic data (Synthetic_data/ subdirectory), and to build the Genomes data set (Genomes/ subdirectory).


Synthetic data
	We advise to use a sufficiently well equipped server to build the synthetic data (>64GB RAM), the resulting data sets will occupy ~4GB of disk space.
	Running the "genSynth.sh" script will generate the data for each set of experiments.

	Note that for each set of experiments, there are two versions of data generated. In all our experiments, we use the data with the _itemOverlap.dat" suffix as input, modeling the more challenging setup where patterns can share individual items (e.g. there are patterns ABC and CDE present in the data, sharing C). The ground truth set of patterns is given by the corresponding file with an additional "_patterns.txt" suffix.

	We provide a python script "datToHDF5.py" to convert .dat files to hdf5 file format, required by the Asso algorithm.
	To generate the synthetic data also in .h5 format, please comment in the corresponding lines in the genSynth.sh script, note however that this will take a significant amount of disk space.

	Adhering to the specific output file formats of each algorithm, we also provide a python script to compute the F1 score between an algorithm output and the ground truth pattern set.
	For more information on how to use the script, call
		python comp_patterns.py -h


Real data
	DNA -- we provide a binarized version of this data as dna_amplification.dat
	Kosarak -- this data is available through the FIMI database http://fimi.uantwerpen.be/data/
	Accidents -- also available through FIMI
	Instacart -- the original data is available through Kaggle (https://www.kaggle.com/c/instacart-market-basket-analysis/data) and we provide the data with merged items upon request (we cannot make it directly available due to copyright).

	Genomes
		For the 1000Genomes data set, we processed the variant calls of the individuals available in phase 3 of the 1000genomes project, mapping them to genes of the corresponding reference genome.

		To generate this dataset, please follow these instructions carefully. Note that this is a large genomic dataset and hence the scripts require sufficient RAM (~1TB) and disk space (~30GB).
		The scripts are written in R and require the "vcfR" package.

		To download the raw variant call files for all autosomes (chromosome 1-22) and generate corresponding .dat files, execute "./downloadVCF.sh".
		Note that this should be executed in the Genomes subfolder, as it requires additional meta information files deposited in this subfolder (the position of genes on the reference, and feature set sizes).
		The process will take approximately 2 days.

		Besides .dat files and additional meta information per chromosome, this results in a .dat file of variants for the full autosome that can be used for mining
			1kgenomes_variants_af0.01_autosome_genebodyOnly.dat
		a file containing variant ID and description ("genotype") for each feature
			1kgenomes_variants_af0.01_autosome_genebodyOnly.colnames
		a file containing gene names for each feature, identifying in which genes a variant occurs in (if a variant occurs in multiple genes, those are separated by a ":")
			1kgenomes_variants_af0.01_autosome_genebodyOnly.genes
		and a file containing the sample IDs
			1kgenomes_variants_af0.01_autosome_genebodyOnly.rownames
