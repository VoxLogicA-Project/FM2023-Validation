# Experimental validation of FM2022 submission: "Minimisation of Spatial Models using Branching Bisimilarity"

## COPYRIGHT

All rights reserved on any file in this archive. Due to double blind submission,
proper attribution cannot be made explicit here. Licensing matters will be
cleaned up before publication.

## Disclaimer

This archives only works on linux (this will be fixed before publication), and
has been specifically tested on ubuntu-20.04 and ubuntu-21.10. This archive does
NOT yet work on ubuntu-22.04 because both GraphLogicA and VoxLogicA need to be
recompiled against libssl3. Will be fixed soon.

Before publication we will consider making this archive portable to more
operating systems (the tools that we use are multiplatform, it is just
reproduction of the experiments that has been prepared on, and for, linux, also
because of the Docker image used). This would however make the archive much
larger.

## Prerequisites

It is very likely that you already have all the dependencies installed, but just in
case: on ubuntu, one needs to install python3 and python3-pip

    sudo apt install python3 python3-pip 
    
after which the python prerequisites pandas and pillow can be installed using 

    pip3 install pandas pillow

## Running the tests

To run the tests, run the file runTests.py either directly

    ./runTests.py

(requires the execute bit set, but it should already be so after unpacking), or
via python3

    python3 ./runTests.py

## Results

See the comments in runTests.py or the paper to see what the tests are doing. 

The results are the files "rawdata.csv" (raw data from experiments) and
"results-table.csv" (table massaged to be included in latex; note that the
labels are not exactly the same as those in the paper but they should be clear
nevertheless.).

## Dockerfile

We also include a Dockerfile building against the ubuntu-20.04 image for
long-term reproducibility. 

Should you be unfamiliar with docker, but willing to use the images, you need to
have docker installed and working, then to build the image, then to run it, for
instance, using the two commands below.

    docker build --progress=plain .
    docker run --rm $(docker build -q .)

We will consider providing a pre-built image for publication.
