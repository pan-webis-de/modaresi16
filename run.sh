#!/bin/bash

OPTIND= # Reset in case getopts has been used previously in the shell.
# Initialize the variables
INPUT=""
RUN=""
OUTPUT=""
while getopts ":c:o" opt; do
  case $opt in
    c) INPUT=$OPTARG
    ;;
    o) OUTPUT=$OPTARG
    ;;
  esac
done
shift $((OPTIND-1))

echo "************************"
echo "*** Starting execution ***"
echo "************************"
echo input    = "${INPUT}"
echo output   = "${OUTPUT}"
 
make build
VOLUMES=" -v $INPUT:/media/input -v $OUTPUT:/media/output  "
VARS=" -e TIRA_INPUT=/media/input -e TIRA_OUTPUT=/media/output "
IMAGE=profiler16_un
CMD=""
name=profiler16_un
registry=hub.docker.com
echo "[BEGIN DOCKER COMMAND]"
docker run -it --rm=true $VOLUMES $VARS $registry/$name python profiler.py --tira_input /media/input --tira_output /media/output
docker run -it --rm=true $VOLUMES $VARS $registry/$name chown -R 1000:1000 /media/output
echo "[END DOCKER COMMAND]"
