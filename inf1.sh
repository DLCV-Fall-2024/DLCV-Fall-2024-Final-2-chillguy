#---------------------------------------------anime-------------------------------------------

# anime_character=0

# if [ ${anime_character} -eq 1 ]
# then
#   fused_model="experiments/composed_edlora/anythingv4/hina+kario+tezuka_anythingv4/combined_model_base"
#   expdir="hina+kario+tezuka_anythingv4"

#   keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose_2x/hina_tezuka_kario_2x.png'
#   keypose_adaptor_weight=1.0
#   sketch_condition=''
#   sketch_adaptor_weight=1.0

#   context_prompt='two girls and a boy are standing near a forest'
#   context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

#   region1_prompt='[a <hina1> <hina2>, standing near a forest]'
#   region1_neg_prompt="[${context_neg_prompt}]"
#   region1='[12, 36, 1024, 600]'

#   region2_prompt='[a <tezuka1> <tezuka2>, standing near a forest]'
#   region2_neg_prompt="[${context_neg_prompt}]"
#   region2='[18, 696, 1024, 1180]'

#   region5_prompt='[a <kaori1> <kaori2>, standing near a forest]'
#   region5_neg_prompt="[${context_neg_prompt}]"
#   region5='[142, 1259, 1024, 1956]'

#   prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region5_prompt}-*-${region5_neg_prompt}-*-${region5}"

#   python regionally_controlable_sampling.py \
#     --pretrained_model=${fused_model} \
#     --sketch_adaptor_weight=${sketch_adaptor_weight}\
#     --sketch_condition=${sketch_condition} \
#     --keypose_adaptor_weight=${keypose_adaptor_weight}\
#     --keypose_condition=${keypose_condition} \
#     --save_dir="results/multi-concept/${expdir}" \
#     --prompt="${context_prompt}" \
#     --negative_prompt="${context_neg_prompt}" \
#     --prompt_rewrite="${prompt_rewrite}" \
#     --suffix="baseline" \
#     --seed=19
# fi

#---------------------------------------------real-------------------------------------------
export CUDA_VISIBLE_DEVICES=1
real_character=1

if [ ${real_character} -eq 1 ]
then
  fused_model="experiments/prompt1/combined_model_base"
  expdir="cat2+dog6_chilloutmix"

  keypose_condition=''
  #keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition=''
  #sketch_condition='datasets/validation_spatial_condition/characters-objects/dog_cat_pose.png'
  sketch_adaptor_weight=1.0

  # context_prompt='three people near the castle, 4K, high quality, high resolution, best quality'
  # context_prompt='a <potter1> <potter2>, a <hermione1> <hermione2> and a <thanos1> <thanos2> near the castle, 4K, high quality, high resolution, best quality'
  context_prompt='A cat on the right and a dog on the left'
  # context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,fewer digits, cropped, worst quality, low quality'
  context_neg_prompt='longbody, lowres, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality'

  # region1_prompt='[a <potter1> <potter2>, in Hogwarts uniform, holding hands, near the castle, 4K, high quality, high resolution, best quality]'
  region1_prompt='[a <cat1> <cat2> on the right of the image]'
  region1_neg_prompt="[${context_neg_prompt}]"
  # region1='[4, 6, 1024, 490]'
  # in the image size 512x512
  #region1='[2, 1302, 1024, 1992]'
  # region1='[4, 800, 512, 1992]'
  region1='[4, 550, 1024, 1000]' #1024x1024 best
  #region1='[30, 791, 555, 991]' # with sketch
  #region1='[4, 333, 208, 490]' # with pose
  # region2_prompt='[a <hermione1> <hermione2>, girl, in Hogwarts uniform, near the castle, 4K, high quality, high resolution, best quality]'
  # region2_prompt='[a <hermione1> <hermione2>,  girl, in Hogwarts uniform, near the mountaion, 4K, high quality, high resolution, best quality]'

  # region2_neg_prompt="[${context_neg_prompt}]"
  # region2='[14, 490, 1024, 920]'
  region2_prompt='[a <dog1> <dog2>, on the left of the image]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[4, 6, 1024, 490]'  # 1024x1024 best
  #region2="[30, 70, 601, 228]"  # with sketch
  #region2='[7, 56, 229, 186]'  # with pose
  # region2='[2, 1302, 1024, 1992]'
  #region2='[4, 6, 1024, 490]'
  # region3_prompt='[a <thanos1> <thanos2>, purple armor, near the castle, 4K, high quality, high resolution, best quality]'
  # region3_prompt='[a <thanos1> <thanos2>, purple armor, near the mountain, 4K, high quality, high resolution, best quality]'
  # region3_neg_prompt="[${context_neg_prompt}]"
  # region3='[2, 1302, 1024, 1992]'
  seed_list='[14, 15, 16, 17, 18, 19, 20, 21, 22, 23]' # for dog6_cat2 
  # prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"
  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"
  # python regionally_controlable_sampling.py \
  #   --pretrained_model=${fused_model} \
  #   --sketch_adaptor_weight=${sketch_adaptor_weight}\
  #   --sketch_condition=${sketch_condition} \
  #   --keypose_adaptor_weight=${keypose_adaptor_weight}\
  #   --keypose_condition=${keypose_condition} \
  #   --save_dir="results/multi-concept/${expdir}" \
  #   --prompt="${context_prompt}" \
  #   --negative_prompt="${context_neg_prompt}" \
  #   --suffix="baseline" \
  #   --seed=14
  python3 inf1.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/${expdir}" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=14
fi
 



