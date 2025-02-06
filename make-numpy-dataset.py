# %%
import pdb

config_blob = """{
    "name": "czii_cryoet_mlchallenge_2024",
    "description": "2024 CZII CryoET ML Challenge training data.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "apo-ferritin",
            "is_particle": true,
            "pdb_id": "4V1W",
            "label": 1,
            "color": [  0, 117, 220, 128],
            "radius": 80,
            "map_threshold": 0.0418
        },
        {
            "name": "beta-amylase",
            "is_particle": true,
            "pdb_id": "1FA2",
            "label": 2,
            "color": [153,  63,   0, 128],
            "radius": 65,
            "map_threshold": 0.035
        },
        {
            "name": "beta-galactosidase",
            "is_particle": true,
            "pdb_id": "6X1Q",
            "label": 3,
            "color": [ 76,   0,  92, 128],
            "radius": 90,
            "map_threshold": 0.0578
        },
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "6EK0",
            "label": 4,
            "color": [  0,  92,  49, 128],
            "radius": 150,
            "map_threshold": 0.0374
        },
        {
            "name": "thyroglobulin",
            "is_particle": true,
            "pdb_id": "6SCJ",
            "label": 5,
            "color": [ 43, 206,  72, 128],
            "radius": 120,
            "map_threshold": 0.0278
        },
        {
            "name": "virus-like-particle",
            "is_particle": true,
            "pdb_id": "6N4V",            
            "label": 6,
            "color": [255, 204, 153, 128],
            "radius": 150,
            "map_threshold": 0.201
        }
    ],

    "overlay_root": "./working/overlay",

    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "./data/train/static"
}"""

copick_config_path = "./working/copick.config"
output_overlay = "./working/overlay"
import os

os.makedirs("./working/", exist_ok=True)
with open(copick_config_path, "w") as f:
    f.write(config_blob)

# %%
# Update the overlay
# Define source and destination directories
source_dir = "./data/train/overlay"
destination_dir = "./working/overlay"

# %%
# Make a copick project
import os
import shutil

# Walk through the source directory
for root, dirs, files in os.walk(source_dir):
    # Create corresponding subdirectories in the destination
    relative_path = os.path.relpath(root, source_dir)
    target_dir = os.path.join(destination_dir, relative_path)
    os.makedirs(target_dir, exist_ok=True)

    # Copy and rename each file
    for file in files:
        if file.startswith("curation_0_"):
            new_filename = file
        else:
            new_filename = f"curation_0_{file}"

        # Define full paths for the source and destination files
        source_file = os.path.join(root, file)
        destination_file = os.path.join(target_dir, new_filename)

        # Copy the file with the new name
        shutil.copy2(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")

# %%
import os
import numpy as np
from pathlib import Path
import torch
import torchinfo
import zarr, copick
from tqdm import tqdm
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    RandFlipd,
    RandRotate90d,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
import mlflow
import mlflow.pytorch

# %%
tomo_type_list = ["ctfdeconvolved", "denoised", "isonetcorrected", "wbp"]

# %%
# root = copick.from_file(copick_config_path)

copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
# tomo_type = "denoised"

# %%
from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
from collections import defaultdict


# Just do this once
generate_masks = True
for tomo_type in tomo_type_list:
    root = copick.from_file(copick_config_path)
    if generate_masks:
        target_objects = defaultdict(dict)
        for object in root.pickable_objects:
            if object.is_particle:
                target_objects[object.name]["label"] = object.label
                target_objects[object.name]["radius"] = object.radius

        for run in tqdm(root.runs):
            tomo = run.get_voxel_spacing(10)
            tomo = tomo.get_tomogram(tomo_type).numpy()
            target = np.zeros(tomo.shape, dtype=np.uint8)
            for pickable_object in root.pickable_objects:
                pick = run.get_picks(
                    object_name=pickable_object.name, user_id="curation"
                )
                # pdb.set_trace()
                if len(pick):
                    # pdb.set_trace()
                    target = segmentation_from_picks.from_picks(
                        pick[0],
                        target,
                        target_objects[pickable_object.name]["radius"] * 0.4,
                        target_objects[pickable_object.name]["label"],
                    )
            write.segmentation(
                run, target, copick_user_name, name=copick_segmentation_name
            )

    data_dicts = []
    dir_name = "./numpy-data-types-point-C"
    os.makedirs(dir_name, exist_ok=True)
    for run in tqdm(root.runs):
        tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type).numpy()
        tomogram_rot90 = np.rot90(tomogram, axes=(1, 2))
        segmentation = run.get_segmentations(
            name=copick_segmentation_name,
            user_id=copick_user_name,
            voxel_size=voxel_size,
            is_multilabel=True,
        )[0].numpy()
        segmentation_rot90 = np.rot90(segmentation, axes=(1, 2))
        data_dicts.append(
            {
                "name": run.name,
                "image": tomogram,
                "image-rot90": tomogram_rot90,
                "label": segmentation,
                "label-rot90": segmentation_rot90,
                "tomo_type": tomo_type,
            }
        )

    for i in range(7):
        with open(
            f"{dir_name}/train_image_{data_dicts[i]['name']}_{data_dicts[i]['tomo_type']}.npy",
            "wb",
        ) as f:
            np.save(f, data_dicts[i]["image"])
        with open(
            f"{dir_name}/train_image_{data_dicts[i]['name']}_{data_dicts[i]['tomo_type']}-rot90.npy",
            "wb",
        ) as f:
            np.save(f, data_dicts[i]["image-rot90"])

        with open(
            f"{dir_name}/train_label_{data_dicts[i]['name']}_{data_dicts[i]['tomo_type']}.npy",
            "wb",
        ) as f:
            # pdb.set_trace()
            np.save(f, data_dicts[i]["label"])
        with open(
            f"{dir_name}/train_label_{data_dicts[i]['name']}_{data_dicts[i]['tomo_type']}-rot90.npy",
            "wb",
        ) as f:
            np.save(f, data_dicts[i]["label-rot90"])
