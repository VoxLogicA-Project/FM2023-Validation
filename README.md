# Experimental validation of FM2022 submission: "Minimisation of Spatial Models using Branching Bisimilarity"

All rights reserved on any file in this archive. Due to double blind submission, proper attribution cannot be made explicit here. Licensing matters will be cleaned up before publication.

Tested on linux only (this will be fixed before publication), ubuntu-20.04 and ubuntu-21.10. 

NOTE: this does NOT work on ubuntu-22.04 because both GraphLogicA and VoxLogicA need to be recompiled against libssl3.

## Running the tests

To run the tests, run the file runTests.py either directly, with the execute bit set, or via python3.

## Prerequisites

On ubuntu, one needs to install python3 and python3-pip

    sudo apt install python3 python3-pip 
    
after which the python prerequisites can be installed using 

    pip3 install pandas pillow

## Results

See the comments in runTests.py to see what the tests are doing. 

The results are the files "rawdata.csv" (raw data from experiments) and "results-table.csv" (table massaged to be included in latex; note that the labels are not exactly the same as those in the paper but they should be clear nevertheless.).


## Dockerfile

We also include a Dockerfile building against the ubuntu-20.04 image. Should you be unfamiliar with docker, but willing to use the images, you need docker installed and working, and then launch the two commands at the beginning of "Dockerfile".
