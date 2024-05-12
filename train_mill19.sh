export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/conda/envs/gsplat_mesh/lib:$LD_LIBRARY_PATH

###############################
## Original Scaffold GS
###############################
# DATA_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm
# SPLIT_FOLDER=colmap_aligned/0
# MESH_FILE=$DATA_PATH/$SPLIT_FOLDER/mesh_deci0.75.ply
# GS_TYPE=gs_mesh
# DATA_NAME='MegaMesh'
# MODEL_NAME='ScaffoldMeshGS'

# python train.py --eval -s $DATA_PATH --lod 0 --gpu -1 --voxel_size 0.01 \
#  --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 60_000 \
#   --port 23925 -m outputs/building/baseline_lr40000_update_20000_60000 --update_from 20000 --update_until 60000 \
#   --split_folder $SPLIT_FOLDER --meshes $MESH_FILE --gs_type $GS_TYPE --data_type $DATA_NAME --model_name $MODEL_NAME

###############################
## Original Scaffold GS with MipSplatting
###############################
# DATA_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm
# SPLIT_FOLDER=colmap_aligned/7
# MESH_FILE=$DATA_PATH/$SPLIT_FOLDER/mesh_deci0.75.ply
# GS_TYPE=gs_mesh
# DATA_NAME=MegaMeshMulti
# MODEL_NAME=ScaffoldMeshMipGS
# OUTPUT_FOLDER=outputs/building/ScMeshMipGS_7_res4

# python train_aa.py --eval -s $DATA_PATH --lod 0 --gpu -1 --voxel_size 0.01 \
#  --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 60_000 \
#   --port 23900 -m $OUTPUT_FOLDER --update_from 20000 --update_until 60000 \
#   --split_folder $SPLIT_FOLDER --meshes $MESH_FILE --gs_type $GS_TYPE --data_type $DATA_NAME --model_name $MODEL_NAME \
#   # --warmup --load_iteration 60000


################################
## Scaffold MeshGS with KNN
###############################
# DATA_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm
# SPLIT_FOLDER=colmap_aligned/7
# MESH_FILE=$DATA_PATH/$SPLIT_FOLDER/mesh_deci0.75.ply
# GS_TYPE=gs_mesh
# DATA_NAME=MegaMesh
# OUTPUT_FOLDER=outputs/building/MeshGSKnn_app32
# MODEL_NAME=ScaffoldMeshKNNGS

# python train.py --eval -s $DATA_PATH --lod 0 --gpu -1 --voxel_size 0.01 \
#  --update_init_factor 16 --appearance_dim 32 --ratio 1 --iterations 60_000 \
#   --port 23925 -m $OUTPUT_FOLDER --update_from 20000 --update_until 60000 \
#   --save_iterations 20000 40000 60000 --checkpoint_iterations 20000 40000 60000 \
#   --split_folder $SPLIT_FOLDER --meshes $MESH_FILE --gs_type $GS_TYPE --data_type $DATA_NAME --model_name $MODEL_NAME \
#   --warmup --load_iteration 60000
#   # --start_checkpoint $OUTPUT_FOLDER/checkpoints/ckpt_20000.pth 


################################
## Scaffold MeshGS with Encoding
###############################
# DATA_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm
# SPLIT_FOLDER=colmap_aligned/7
# MESH_FILE=$DATA_PATH/$SPLIT_FOLDER/mesh_deci0.75.ply
# GS_TYPE=gs_mesh
# DATA_NAME=MegaMesh
# OUTPUT_FOLDER=outputs/building/ScMeshGSwEncoding_7_app0
# MODEL_NAME=ScaffoldMeshGSwEncoding

# python train.py --eval -s $DATA_PATH --lod 0 --gpu -1 --voxel_size 0.01 \
#  --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 60_000 \
#   --port 23925 -m $OUTPUT_FOLDER --update_from 20000 --update_until 60000 \
#   --save_iterations 20000 40000 60000 --checkpoint_iterations 20000 40000 60000 \
#   --split_folder $SPLIT_FOLDER --meshes $MESH_FILE --gs_type $GS_TYPE --data_type $DATA_NAME --model_name $MODEL_NAME \
# #   --warmup --load_iteration 60000
#   # --start_checkpoint $OUTPUT_FOLDER/checkpoints/ckpt_20000.pth 


# DATA_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm
# SPLIT_FOLDER=colmap_aligned/7
# MESH_FILE=$DATA_PATH/$SPLIT_FOLDER/mesh_deci0.75.ply
# GS_TYPE=gs_mesh
# DATA_NAME=MegaMesh
# OUTPUT_FOLDER=outputs/building/MeshGSwEncoding_7_app48
# MODEL_NAME=ScaffoldMeshGSwEncoding

# python train.py --eval -s $DATA_PATH --lod 0 --gpu -1 --voxel_size 0.01 \
#  --update_init_factor 16 --appearance_dim 48 --ratio 1 --iterations 60_000 \
#   --port 23925 -m $OUTPUT_FOLDER --update_from 20000 --update_until 60000 \
#   --save_iterations 20000 40000 60000 --checkpoint_iterations 20000 40000 60000 \
#   --split_folder $SPLIT_FOLDER --meshes $MESH_FILE --gs_type $GS_TYPE --data_type $DATA_NAME --model_name $MODEL_NAME \
# #   --warmup --load_iteration 60000
#   # --start_checkpoint $OUTPUT_FOLDER/checkpoints/ckpt_20000.pth 


###############################
##  Vast Scaffold GS 
###############################
# DATA_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm
# # DATA_PATH=/scratch2/choi/data/mega_nerf_data/building-pixsfm
# SPLIT_FOLDER=colmap_aligned/7
# MESH_FILE=$DATA_PATH/$SPLIT_FOLDER/mesh_deci0.75.ply
# GS_TYPE=gs_mesh
# DATA_NAME=MegaMesh
# MODEL_NAME=VastScaffoldMeshGS
# OUTPUT_FOLDER=outputs/building/VastScMeshGS_7_app0

# python train_app.py --eval -s $DATA_PATH --lod 0 --gpu -1 --voxel_size 0.01 \
#  --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 60_000 \
#   --port 23925 -m $OUTPUT_FOLDER --update_from 20000 --update_until 60000 \
#   --split_folder $SPLIT_FOLDER --meshes $MESH_FILE --gs_type $GS_TYPE --data_type $DATA_NAME --model_name $MODEL_NAME

# DATA_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm
# # DATA_PATH=/scratch2/choi/data/mega_nerf_data/building-pixsfm
# SPLIT_FOLDER=colmap_aligned/7
# MESH_FILE=$DATA_PATH/$SPLIT_FOLDER/mesh_deci0.75.ply
# GS_TYPE=gs_mesh
# DATA_NAME=MegaMesh
# MODEL_NAME=VastScaffoldMeshGS
# OUTPUT_FOLDER=outputs/building/VastScMeshGS_7_app32

# python train_app.py --eval -s $DATA_PATH --lod 0 --gpu -1 --voxel_size 0.01 \
#  --update_init_factor 16 --appearance_dim 32 --ratio 1 --iterations 60_000 \
#   --port 23925 -m $OUTPUT_FOLDER --update_from 20000 --update_until 60000 \
#   --split_folder $SPLIT_FOLDER --meshes $MESH_FILE --gs_type $GS_TYPE --data_type $DATA_NAME --model_name $MODEL_NAME


