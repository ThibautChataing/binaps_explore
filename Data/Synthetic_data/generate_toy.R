require(plyr)


## Introduce additive noise to the database specified by the matrix
## Arguments:
##          m           database to manipulate
##          perc        percentage of 0s to be flipped
add_additive_noise <- function(m, perc)
{
    # determine position of zeroes in matrix
    zero_pos <- which(m == 0)

    # draw positions to flip
    n_flips <- ceiling(perc*length(zero_pos))
    flip_pos <- sample(zero_pos, n_flips)

    # add noise
    sapply(flip_pos, function(i) {m[i] <<- 1})

    return(m)
}

## Introduce destructive noise to the database specified by the matrix
## Arguments:
##          m           database to manipulate
##          perc        percentage of 1s to be flipped
add_destructive_noise <- function(m, perc)
{
    # determine position of ones in matrix
    one_pos <- which(m == 1)

    # draw positions to flip
    n_flips <- ceiling(perc*length(one_pos))
    flip_pos <- sample(one_pos, n_flips)

    # add noise
    sapply(flip_pos, function(i) {m[i] <<- 0})

    return(m)
}


## Introduce noise into database specified by matrix by randomly flipping entries
## Arguments:
##          m           database to manipulate
##          perc        percentage of 1s to be flipped
add_noise <- function(m, perc)
{
	n_cells <- prod(dim(m))
    n_flips <- ceiling(perc*n_cells)
    flip_pos <- sample(1:n_cells, n_flips)

    sapply(flip_pos, function(i) {
    		if (m[i] == 0)
    		{
    			m[i] <<- 1
    		} else {
    			m[i] <<- 0
    		}

    	}

    )
    return(m)
}



genData <- function(npats, patlen, nrows, density, noiselvl)
{

	patCols <- vector("list", length = npats)
	patRows <- vector("list", length = npats)
	nextCol <- 1
	for (pid in 1:npats)
	{
		lnum <- sample(2:patlen, 1)
		# assign feature ids to patterns
		patCols[[pid]] <- nextCol:(nextCol+lnum-1)
		nextCol <- nextCol + lnum

		nr <- ceiling(rnorm(1, density*nrows, density*nrows/10))
		# redraw illegal numbers
		while (nr<=0 || nr >= nrows)
		{
			nr <- ceiling(rnorm(1, density*nrows, density*nrows/10))
		}
		patRows[[pid]] <- sample(nrows,nr)
	}

	ncols <- sum(sapply(patCols, length))
	print(ncols)
	m <- matrix(0, nrows, ncols)
	print(dim(m))

	# apply patterns
	for (pid in 1:npats)
	{
		m[patRows[[pid]], patCols[[pid]] ] <- 1
	}

	m <- add_noise(m, noiselvl)

	#database_raw <- m

	print("Added noise.")

	# convert database to .dat format
	m <- lapply(
	        1:nrow(m),
	        function (idx)
	        {
	            return(as.character(which(m[idx,] == 1)))
	        }
	    )
	print("Converted to dat file.")

	# remove rows without content
	m <- m[lapply(m,length)>0]
	print("Removed rows without content.")
	return(list(data=m, pats=patCols))

}


genDataSharedCols <- function(npats, patlen, nrows, density, noiselvl)
{

	patRows <- vector("list", length = npats)
	patLnum <- vector("list", length = npats)
	#nextCol <- 1
	lmax <- 0
	for (pid in 1:npats)
	{
		patLnum[[pid]] <- sample(2:patlen, 1)
		lmax <- lmax + patLnum[[pid]]

		nr <- ceiling(rnorm(1, density*nrows, density*nrows/10))
		# redraw illegal numbers
		while (nr<=0 || nr >= nrows)
		{
			nr <- ceiling(rnorm(1, density*nrows, density*nrows/10))
		}
		patRows[[pid]] <- sample(nrows,nr)
	}

	patCols <- vector("list", length = npats)
	for (pid in 1:npats)
	{
		patCols[[pid]] <- sample(1:lmax, patLnum[[pid]])
	}

	m <- matrix(0, nrows, lmax)

	# apply patterns
	for (pid in 1:npats)
	{
		m[patRows[[pid]], patCols[[pid]] ] <- 1
	}

	# remove columns not used by any pattern and adapt pattern ids
	nonzeroIdcs <- sapply(1:ncol(m), function(idx){sum(m[,idx]) > 0})
	offsets <- rep(0, ncol(m))
	skipCount <- 0
	for (i in 1:ncol(m))
	{
		offsets[i] <- skipCount
		if (!nonzeroIdcs[i])
		{
			skipCount <- skipCount + 1
		}
	}
	m <- m[, nonzeroIdcs]
	# reindex patterns
	patCols <- lapply(patCols, function(pat)
	{
		sapply(pat, function(idx){idx - offsets[idx]})
	})

	print(dim(m))

	m <- add_noise(m, noiselvl)

	#database_raw <- m

	# convert database to .dat format
	m <- lapply(
	        1:nrow(m),
	        function (idx)
	        {
	            return(as.character(which(m[idx,] == 1)))
	        }
	    )

	# remove rows without content
	m <- m[lapply(m,length)>0]
	return(list(data=m, pats=patCols))

}


pat_to_string <- function (pat)
{
	return(paste(pat, collapse=" "))
}
pats_to_string <- function (pats)
{
	return(lapply(pats, function(pat)
	{
		pat_to_string(pat)
	}))
}




args <- commandArgs(trailingOnly=TRUE)

# Number of patterns to plant in data
npats <- as.numeric(args[2])
# Number of rows in database
nrows <- as.numeric(args[3])
# Maximum number of literals in conjunction
maxLits <- as.numeric(args[4])
# Output file name
output_file <- args[5]
# Level of noise to add
noiseLvl <- as.numeric(args[6])
# density/frequency of patterns
density <- as.numeric(args[7])

datList <- genData(npats, maxLits, nrows, density, noiseLvl)

m <- datList$data
pats <- datList$pats
#raw <- datList$data_raw

# get maximum number of ones in a row
maxCols <- max(
    sapply(1:length(m),
    function (idx)
    {
        return(length(m[[idx]]))
    }
    )
)


output_file1 <- paste(output_file, ".dat", sep="")
# flush output file
close(file(output_file1, open="w"))
# write result to specified file
invisible(lapply(m, write, output_file1, append=TRUE, ncolumns=maxCols, sep=" "))

patStr <- pats_to_string(pats)
writeLines(sapply(patStr, function(p){p}), paste(output_file1,"_patterns.txt",sep=""))

gc()

datList <- genDataSharedCols(npats, maxLits, nrows, density, noiseLvl)

m2 <- datList$data
pats <- datList$pats
#raw <- datList$data_raw

# get maximum number of ones in a row
maxCols <- max(
    sapply(1:length(m2),
    function (idx)
    {
        return(length(m2[[idx]]))
    }
    )
)

output_file2 <- paste(output_file, "_itemOverlap.dat", sep="")
# flush output file
close(file(output_file2, open="w"))
# write result to specified file
invisible(lapply(m2, write, output_file2, append=TRUE, ncolumns=maxCols, sep=" "))

patStr <- pats_to_string(pats)
writeLines(sapply(patStr, function(p){p}), paste(output_file2,"_patterns.txt",sep=""))
