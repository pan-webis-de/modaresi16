#!/bin/bash


while getopts 'o:c:' opt; do
    case $opt in
        o)  OUTPUT="$OPTARG" ;;
        c)  INPUT="$OPTARG"    ;;
        *)  exit 1            ;;
    esac
done

echo "************************"
echo "*** Starting execution ***"
echo "************************"
echo input    = "${INPUT}"
echo output   = "${OUTPUT}"
 
# make build
EN_TRAINING=/media/training-datasets/author-profiling/pan16-author-profiling-training-dataset-english-2016-04-25
ES_TRAINING=/media/training-datasets/author-profiling/pan16-author-profiling-training-dataset-spanish-2016-04-25
NL_TRAINING=/media/training-datasets/author-profiling/pan16-author-profiling-training-dataset-dutch-2016-04-25
VOLUMES=" -v $INPUT:/media/input -v $OUTPUT:/media/output -v $EN_TRAINING:/media/en -v $ES_TRAINING:/media/es -v $NL_TRAINING:/media/nl "
VARS=" -e TIRA_INPUT=/media/input -e TIRA_OUTPUT=/media/output "
IMAGE=profiler16_un
name=profiler16_un
registry=hub.docker.com
echo "[BEGIN DOCKER COMMAND]"
docker run -it --rm=true $VOLUMES $VARS $registry/$name python profiler.py --tira_input /media/input --tira_output /media/output
docker run -it --rm=true $VOLUMES $VARS $registry/$name chown -R 1000:1000 /media/output
echo "[END DOCKER COMMAND]"
