#!/bin/bash

nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
	      -v /home/tkurth/ClimDeepLearn:/mnt/climdeeplearn:ro \
	      -v /data1/tkurth/tiramisu/segm_h5_v3_new_split:/mnt/data:ro \
	      -v /data1/tkurth/tiramisu/runs:/mnt/runs:rw \
	      -v /data1/tkurth/tiramisu/tmp:/mnt/tmp:rw \
	      -w /mnt/climdeeplearn/semanticsegm/run_scripts/maeve \
	      -i -t tensorflow-profile:latest bash #tensorrt:latest bash
