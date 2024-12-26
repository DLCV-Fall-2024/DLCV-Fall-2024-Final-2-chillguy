# fuse real character
if [ $1 -eq 4 ]
then
  config_file="cat2+wearable_glasses+watercolor"
  prompt="4"
#   python3 gradient_fusion.py --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/real/${config_file}.json" --save_path="experiments/composed_edlora/chilloutmix/${config_file}" --pretrained_models="experiments/pretrained_models/chilloutmix" --optimize_textenc_iters=500 --optimize_unet_iters=50
fi

if [ $1 -eq 1 ]
then
  config_file="cat2+dog6"
prompt="1"
#   python3 gradient_fusion.py --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/real/${config_file}.json" --save_path="experiments/composed_edlora/chilloutmix/${config_file}" --pretrained_models="experiments/pretrained_models/chilloutmix" --optimize_textenc_iters=500 --optimize_unet_iters=50
fi

if [ $1 -eq 2 ]
then
  config_file="flower_1+vase"
prompt="2"
#   python3 gradient_fusion.py --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/real/${config_file}.json" --save_path="experiments/composed_edlora/chilloutmix/${config_file}" --pretrained_models="experiments/pretrained_models/chilloutmix" --optimize_textenc_iters=500 --optimize_unet_iters=50
fi

if [ $1 -eq 3 ]
then
  config_file="dog+pet_cat1+dog6"
prompt="3"
#   python3 gradient_fusion.py --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/real/${config_file}.json" --save_path="experiments/composed_edlora/chilloutmix/${config_file}" --pretrained_models="experiments/pretrained_models/chilloutmix" --optimize_textenc_iters=500 --optimize_unet_iters=50
fi
python3 gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/real/${config_file}.json" \
    --save_path="experiments/prompt${prompt}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50

# # fuse anime character
# config_file="hina+kario+tezuka_anythingv4"

# python gradient_fusion.py \
#     --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/anime/${config_file}.json" \
#     --save_path="experiments/composed_edlora/anythingv4/${config_file}" \
#     --pretrained_models="experiments/pretrained_models/anything-v4.0" \
#     --optimize_textenc_iters=500 \
#     --optimize_unet_iters=50
