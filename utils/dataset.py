import pandas as pd
import numpy as np
import json
import zarr

PARTICLE = [
    {
        "name": "apo-ferritin",
        "difficulty": "easy",
        "pdb_id": "4V1W",
        "label": 1,
        "color": [0, 255, 0, 0],
        "radius": 60,
        "map_threshold": 0.0418,
    },
    {
        "name": "beta-amylase",
        "difficulty": "ignore",
        "pdb_id": "1FA2",
        "label": 2,
        "color": [0, 0, 255, 255],
        "radius": 65,
        "map_threshold": 0.035,
    },
    {
        "name": "beta-galactosidase",
        "difficulty": "hard",
        "pdb_id": "6X1Q",
        "label": 3,
        "color": [0, 255, 0, 255],
        "radius": 90,
        "map_threshold": 0.0578,
    },
    {
        "name": "ribosome",
        "difficulty": "easy",
        "pdb_id": "6EK0",
        "label": 4,
        "color": [0, 0, 255, 0],
        "radius": 150,
        "map_threshold": 0.0374,
    },
    {
        "name": "thyroglobulin",
        "difficulty": "hard",
        "pdb_id": "6SCJ",
        "label": 5,
        "color": [0, 255, 255, 0],
        "radius": 130,
        "map_threshold": 0.0278,
    },
    {
        "name": "virus-like-particle",
        "difficulty": "easy",
        "pdb_id": "6N4V",
        "label": 6,
        "color": [0, 0, 0, 255],
        "radius": 135,
        "map_threshold": 0.201,
    },
]

PARTICLE_COLOR = [[0, 0, 0]] + [PARTICLE[i]["color"][1:] for i in range(6)]
PARTICLE_NAME = ["none"] + [PARTICLE[i]["name"] for i in range(6)]

"""
(184, 630, 630)  
(92, 315, 315)  
(46, 158, 158)  
"""


def read_one_data(id, static_dir):
    zarr_dir = f"{static_dir}/{id}/VoxelSpacing10.000"
    zarr_file = f"{zarr_dir}/denoised.zarr"
    zarr_data = zarr.open(zarr_file, mode="r")
    volume = zarr_data[0][:]
    max = volume.max()
    min = volume.min()
    volume = (volume - min) / (max - min)
    volume = volume.astype(np.float16)
    return volume


def read_one_truth(id, overlay_dir):
    location = {}

    json_dir = f"{overlay_dir}/{id}/Picks"
    for p in PARTICLE_NAME[1:]:
        json_file = f"{json_dir}/{p}.json"

        with open(json_file, "r") as f:
            json_data = json.load(f)

        num_point = len(json_data["points"])
        loc = np.array(
            [
                list(json_data["points"][i]["location"].values())
                for i in range(num_point)
            ]
        )
        location[p] = loc

    return location
