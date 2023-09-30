# DSTSS
The extend version and all codes of the paper, Distributed Semantic Trajectory Similarity Search.

# Paper of Complete Version
For a better view of the complete version of the paper, please download the pdf at [DSTSS-complete-version](https://github.com/ZJU-DAILY/DSTSS/blob/main/DSTSS-complete-version.pdf).


## Code Introduction
  This folder (i.e., ./at2vec, ./dist-bfjoin-maven, ./hnsw-spatial) holds the source codes in our paper, Distributed Semantic Trajectory Similarity Search.


## Environment Preparation
  - Spark version: 3.1.1
  - A cluster containing 6 nodes, where each node is equipped with two 12-core processors (Intel Xeon E-5-2620 v3 2.40 GHz), 64GB RAM
  - System version: Ubuntu 14.04.3 LTS
  - Java version: 1.8.0
  - Please refer to the source code to install all required packages of libs by Maven.


## Dataset Description
  - "./TrajectorySimCal/src/main/resources/test_data.txt" contains one of the tested datasets (TDrive)
  - The format of the tested dataset is:
         Moving object ID, point ID, location of x, location of y, SR vector of $v_1$, SA vector of $v_2$ 
  
  Note: The synthetic dataset (Brinkhoff) can refer to http://iapg.jade-hs.de/personen/brinkhoff/generator/
  

## Running 
  - Select one of the three projects (./at2vec, ./dist-bfjoin-maven, ./hnsw-spatial);
  - Import the selected project to Intellij IDEA；
  - Downloading all required dependences by Maven; 
  - Initial the configuration file (“./PROJECT_NAME/src/main/resources/pom.xml");
  - Package the project to a X.jar, where X is your project name；
  - Load the packaged X.jar to the master node of your Spark cluster；
  - Running your project in a cluster environment.

