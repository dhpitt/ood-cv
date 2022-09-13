#!/bin/bash
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT=$(get_abs_filename "./")

ENABLE_OCCLUDED=false
DATAROOT="/research/cwloka/projects/dpitt/data"
PATH_ROBIN="${DATAROOT}/ROBINv1.1/ROBINv1.1"

PATH_CACHE_TRAINING_SET="${DATAROOT}/ROBIN_train_NeMo/"
PATH_CACHE_TESTING_SET="${DATAROOT}/ROBIN_NeMo/"
PATH_CACHE_TESTING_SET_OCC="${DATAROOT}/ROBIN_OCC_NeMo/"

OCC_LEVELS=("FGL1_BGL1"  "FGL2_BGL2"  "FGL3_BGL3")
MESH_DIMENSIONS=("single"  "multi")

####################################################################################################
# Process meshes
for MESH_D in "${MESH_DIMENSIONS[@]}"; do
    python ./tools/CreateCuboidMesh.py --CAD_path "${PATH_ROBIN}" --mesh_d "${MESH_D}"
done


####################################################################################################
# Create 3D annotations
for MESH_D in "${MESH_DIMENSIONS[@]}"; do
    python ./code/dataset/generate_3Dpascal3D.py --overwrite False \
            --root_path "${PATH_CACHE_TRAINING_SET}" --mesh_path "${PATH_ROBIN}" --mesh_d "${MESH_D}" &
    python ./code/dataset/generate_3Dpascal3D.py --overwrite False \
            --root_path "${PATH_CACHE_TESTING_SET}" --mesh_path "${PATH_ROBIN}" --mesh_d "${MESH_D}" &
done

wait

####################################################################################################
# Link 3D annotations to occluded datasets
if ${ENABLE_OCCLUDED}; then
    for MESH_D in "${MESH_DIMENSIONS[@]}"; do
        for OCC_LEVEL in "${OCC_LEVELS[@]}"; do
            python ./code/dataset/link_annotations.py --source_path "${PATH_CACHE_TESTING_SET}" --target_path "${PATH_CACHE_TESTING_SET_OCC}" \
                    --occ_level "${OCC_LEVEL}" --mesh_d "${MESH_D}"
            python ./code/dataset/refine_list.py --root_path "${PATH_CACHE_TESTING_SET_OCC}" \
                    --occ_level "${OCC_LEVEL}" --mesh_d "${MESH_D}"
        done
    done
fi 






