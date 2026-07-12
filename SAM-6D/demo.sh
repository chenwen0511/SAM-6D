# Resolve paths: sam6d_env exports SAM6D_CAD_PATH; README uses CAD_PATH
CAD_PATH="${CAD_PATH:-${SAM6D_CAD_PATH:-Data/Example/obj_000005.ply}}"
RGB_PATH="${RGB_PATH:-Data/Example/rgb.png}"
DEPTH_PATH="${DEPTH_PATH:-Data/Example/depth.png}"
CAMERA_PATH="${CAMERA_PATH:-Data/Example/camera.json}"
OUTPUT_DIR="${OUTPUT_DIR:-Data/Example/outputs}"

for var in CAD_PATH RGB_PATH DEPTH_PATH CAMERA_PATH; do
    eval "path=\$$var"
    if [ ! -f "$path" ]; then
        echo "Error: $var not found: $path"
        exit 1
    fi
done
mkdir -p "$OUTPUT_DIR"

# Render CAD templates
cd Render
blenderproc run render_custom_templates.py --output_dir "$OUTPUT_DIR" --cad_path "$CAD_PATH" #--colorize True 


# Run instance segmentation model
export SEGMENTOR_MODEL=sam

cd ../Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir "$OUTPUT_DIR" --cad_path "$CAD_PATH" --rgb_path "$RGB_PATH" --depth_path "$DEPTH_PATH" --cam_path "$CAMERA_PATH"


# Run pose estimation model
export SEG_PATH="$OUTPUT_DIR/sam6d_results/detection_ism.json"

cd ../Pose_Estimation_Model
python run_inference_custom.py --output_dir "$OUTPUT_DIR" --cad_path "$CAD_PATH" --rgb_path "$RGB_PATH" --depth_path "$DEPTH_PATH" --cam_path "$CAMERA_PATH" --seg_path "$SEG_PATH"

