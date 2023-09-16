from pytorch import init, inference
import pathlib
import os

NAME="car"
TARGET="ambulance car"
print(NAME, TARGET)
BASE_DIR = pathlib.Path(os.getcwd())  # YOUR_BASE_DIR
CONFIG: dict  = {
    "BASE_DIR": BASE_DIR,
    "GET3D_PATH": BASE_DIR / "GET3D",  # Path to GET3D (for import and model initialization)
    "TORCH_ENABLED": True,  # False disables all torch operations
    "TORCH_LOG_LEVEL": 2,  # 0: silent, 1: call, 2: 1 + process, 3: 2 + nada output
    "TORCH_WARM_UP_ITER": 10,  # Number of warm up iterations
    "TORCH_WITHOUT_CUSTOM_OPS_COMPILE": False,  # True disables custom c++ ops, instead use pytorch impl
    "TORCH_DEVICE": "cuda:0",  # Device to use
    "NADA_WEIGHT_DIR": BASE_DIR / "weights/get3d_nada",  # Path of NADA weights
    "CLIP_MAP_PATH": BASE_DIR / "weights/clip_map/checkpoint_group.pt",  # Path of embedding group
    "MODEL_OPTS": {  # Model initialization kwargs which is compatible with script arguments
        'latent_dim': 512,
        'one_3d_generator': True,
        'deformation_multiplier': 1.,
        'use_style_mixing': True,
        'dmtet_scale': 1.,
        'feat_channel': 16,
        'mlp_latent_channel': 32,
        'tri_plane_resolution': 256,
        'n_views': 1,
        'render_type': 'neural_render',  # or 'spherical_gaussian'
        'use_tri_plane': True,
        'tet_res': 90,
        'geometry_type': 'conv3d',
        'data_camera_mode': 'shapenet_car',
        'n_implicit_layer': 1,
        'cbase': 32768,
        'cmax': 512,
        'fp32': False
    },
    "TORCH_SEED": 0,  # Random seed
    "TORCH_RESOLUTION": 1024  # Resolution of the output image
}
init(CONFIG)

result = inference(NAME, TARGET)
os.makedirs("exports", exist_ok=True)
with open(f"exports/{NAME}-{TARGET}.glb", "wb") as fp:
    fp.write(result.file.getvalue())  # result.file: BytesIO

with open(f"exports/{NAME}-{TARGET}-thumbnail.png", "wb") as fp:
    fp.write(result.thumbnail.getvalue())  # result.thumbnail: BytesIO