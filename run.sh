#!/bin/bash
# Use > 1 to consume two arguments per pass in the loop (e.g. each
# argument has a corresponding value to go with it).
# Use > 0 to consume one or more arguments per pass in the loop (e.g.
# some arguments don't have a corresponding value to go with it such
# as in the --default example).
# note: if this is set to > 0 the /etc/hosts part is not recognized ( may be a bug )
while [[ $# > 1 ]]
do
key="$1"

case $key in
    -c|--corpus)
    INPUT="$2"
    shift # past argument
    ;;
    -r|--run)
    RUN="$2"
    shift # past argument
    ;;
    -o|--output)
    OUTPUT="$2"
    shift # past argument
    ;;
    --default)
    DEFAULT=YES
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done
echo "************************"
echo "*** Tira test runner ***"
echo "************************"
echo Host input    = "${INPUT}"
echo Host output   = "${OUTPUT}"
echo Host run      = "${RUN}"
 
make build
VOLUMES=" -v $INPUT:/media/input -v $OUTPUT:/media/output  "
VARS=" -e TIRA_INPUT=/media/input -e TIRA_OUTPUT=/media/output "
IMAGE=profiler16_un
CMD=""
echo "[BEGIN DOCKER COMMAND]"
docker run -it --rm=true $VOLUMES $VARS $IMAGE python profiler --tira_input /media/input --tira_output /media/output
docker run -it --rm=true $VOLUMES $VARS $IMAGE chown -R 1000:1000 /media/output
echo "[END DOCKER COMMAND]"
