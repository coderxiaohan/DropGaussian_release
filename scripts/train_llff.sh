exp_name='llff3'
scenes=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
dataset_path='/home/wuxiaohan/wxh/data/nerf_llff_data'
n_views=3

for scene in "${scenes[@]}"
do
  echo "Training on $scene..."
  python train.py -s $dataset_path/$scene/ \
    -m output/raw/$exp_name/$scene \
    --eval -r 8 \
    --n_views $n_views
  
  echo "Rendering $scene..."
  python render.py -m output/raw/$exp_name/$scene -r 8
done

# Compute metrics for all scenes
python metric.py --path output/raw/$exp_name
