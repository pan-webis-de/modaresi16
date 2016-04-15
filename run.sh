#!/bin/bash
while getopts ":c:r:o" opt; do
  case $opt in
    c) INPUT="$OPTARG"
    ;;
    r) RUN="$OPTORG"
    ;;
    o) OUTPUT="$OPTARG"
    ;;
  esac
done

echo "************************"
echo "*** Starting execution ***"
echo "************************"
echo input    = "${INPUT}"
echo output   = "${OUTPUT}"
echo run      = "${RUN}"
 
make build
VOLUMES=" -v $INPUT:/media/input -v $OUTPUT:/media/output  "
VARS=" -e TIRA_INPUT=/media/input -e TIRA_OUTPUT=/media/output "
IMAGE=profiler16_un
CMD=""
echo "[BEGIN DOCKER COMMAND]"
docker run -it --rm=true $VOLUMES $VARS $IMAGE python profiler --tira_input /media/input --tira_output /media/output
docker run -it --rm=true $VOLUMES $VARS $IMAGE chown -R 1000:1000 /media/output
echo "[END DOCKER COMMAND]"
