#!/usr/bin/python3

import argparse
import glob
import csv
import statistics

arg_parser = argparse.ArgumentParser("Takes a selection of scene files and returns statistics regarding the scene length")
arg_parser.add_argument("scene_file_path_wildcard")
arg_parser.add_argument("--sampling-frequency", type=float, default=10.0)
args = arg_parser.parse_args()

scene_lengths = []

for scene_file_path in glob.glob(args.scene_file_path_wildcard):
    with open(scene_file_path, 'r') as scene_file:
        time_step_count = 0

        scene_file_reader = csv.reader(scene_file)
        for row in scene_file_reader:
            time_step_count += 1

        scene_length = time_step_count / args.sampling_frequency

        scene_lengths.append(scene_length)

scene_length_mean = statistics.mean(scene_lengths)
scene_length_stdev = statistics.stdev(scene_lengths)

print(f"Scene Length Mean: {scene_length_mean} s")
print(f"Scene Length Std. Dev.: {scene_length_stdev} s")
