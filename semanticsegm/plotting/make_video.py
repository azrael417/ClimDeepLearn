import os
from glob import glob
import argparse


if __name__ == "__main__":
	AP = argparse.ArgumentParser()
	AP.add_argument("--imagespath",type=str,default=None,required=True,help="Path in which to find the images")
	AP.add_argument("--mode", type=str, default='predictions',help="Either 'prediction' or 'label', corresponding to making a video of the predictions or the labels")
	AP.add_argument("--outputpath",type=str,default=None,required=True,help="Path at which to output the video")
	AP.add_argument("--year",type=int,required=True,help="The year for which to generate the output")
	parsed_args = AP.parse_args()

	ffmpeg_command = "ffmpeg -framerate 10 -pattern_type glob -i '{}*{}*.png' -c:v libx264 -pix_fmt yuv420p {}TMQ_movie.mp4".format(parsed_args.imagespath,
		parsed_args.mode, parsed_args.outputpath)

	os.system(ffmpeg_command)