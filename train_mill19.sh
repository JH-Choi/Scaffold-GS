export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/conda/envs/gsplat_mesh/lib:$LD_LIBRARY_PATH

# Original Scaffold GS
DATA_PATH=/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm
SPLIT_FOLDER=colmap_aligned/0
MESH_FILE=$DATA_PATH/$SPLIT_FOLDER/mesh_deci0.75.ply
GS_TYPE=gs_mesh
DATA_NAME='MegaMesh'
MODEL_NAME='ScaffoldMeshGS'

python train.py --eval -s $DATA_PATH --lod 0 --gpu -1 --voxel_size 0.01 \
 --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations 60_000 \
  --port 23925 -m outputs/building/baseline_lr40000_update_20000_60000 --update_from 20000 --update_until 60000 \
  --split_folder $SPLIT_FOLDER --meshes $MESH_FILE --gs_type $GS_TYPE --data_type $DATA_NAME --model_name $MODEL_NAME