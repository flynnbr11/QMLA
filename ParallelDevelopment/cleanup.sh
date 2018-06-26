#!/bin/bash

mv *.o* *.e* oldBCresults/

mkdir -p oldBCresults/logs
mkdir -p oldBCresults/logs/OUTPUT_ERROR_FILES/

mv logs/OUTPUT_ERROR_FILES/* oldBCresults/logs/OUTPUT_ERROR_FILES/
mv logs/* oldBCresults/logs/
