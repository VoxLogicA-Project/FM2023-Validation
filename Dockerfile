# syntax=docker/dockerfile:1

# TO BUILD THE IMAGE AND VIEW THE PROGRESS:
# docker build --progress=plain . 
# TO RUN THE IMAGE: 
# docker run --rm $(docker build -q .)

FROM ubuntu:20.04

RUN apt update
RUN apt dist-upgrade
RUN apt install -y python3-pip python3-dev unzip 

RUN pip3 install --upgrade pip
RUN pip3 install pandas pillow

RUN useradd MinimizeImages
WORKDIR /home/MinimizeImages
COPY tools ./tools
COPY test-images ./test-images
RUN chown MinimizeImages . -R

ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

USER MinimizeImages

COPY runTests.py . 
CMD python3 ./runTests.py


