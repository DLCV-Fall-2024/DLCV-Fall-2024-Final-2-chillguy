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

real_character=1
export CUDA_VISIBLE_DEVICES=1
if [ ${real_character} -eq 1 ]
then
  fused_model="experiments/prompt3/combined_model_base"
  expdir="dog_pet_cat1_dog6_chilloutmix"

  #keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose_2x/harry_hermione_thanos_2x.png'
  #keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/bengio_lecun_bengio.png'
  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition=''
  #sketch_condition='datasets/validation_spatial_condition/characters-objects/dog6+pet_cat+dog.png'
  sketch_adaptor_weight=1.0
  # A <flower_1> in a <vase>.
  # context_prompt='three people near the castle, 4K, high quality, high resolution, best quality'
  # context_prompt='a <potter1> <potter2>, a <hermione1> <hermione2> and a <thanos1> <thanos2> near the castle, 4K, high quality, high resolution, best quality'
 # context_prompt='three people near the castle, 4K, high quality, high resolution, best quality'
  # context_prompt='a <potter1> <potter2>, a <hermione1> <hermione2> and a <thanos1> <thanos2> near the castle, 4K, high quality, high resolution, best quality'
  # context_prompt='A cat on the right and a dog on the left'
  # context_prompt='A <dog1> <dog2>, a <pet_cat1> <pet_cat2> and a <dog12> <dog22> near a forest'
  context_prompt='two dogs and one cat near a forest'
  # context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,fewer digits, cropped, worst quality, low quality'
  context_neg_prompt='extra digit, fewer digits, cropped, worst quality, low quality'

  # region1_prompt='[a <potter1> <potter2>, in Hogwarts uniform, holding hands, near the castle, 4K, high quality, high resolution, best quality]'
  region1_prompt='[a <dog1> <dog2>, in the left]'
  region1_neg_prompt="[${context_neg_prompt}]"
  # region1='[4, 6, 1024, 490]'
  # in the image size 512x512

  region1='[4, 0, 512, 300]' # no keypose no sketch
  #region1='[230, 100, 825, 324]'

  # region2_prompt='[a <hermione1> <hermione2>, girl, in Hogwarts uniform, near the castle, 4K, high quality, high resolution, best quality]'
  # region2_prompt='[a <hermione1> <hermione2>,  girl, in Hogwarts uniform, near the mountaion, 4K, high quality, high resolution, best quality]'

  # region2_neg_prompt="[${context_neg_prompt}]"
  # region2='[14, 490, 1024, 920]'
  region2_prompt='[a <cat3> <cat4>, in the middle]'
  region2_neg_prompt="[${context_neg_prompt}]"
  #region2='[4, 370, 400, 650]'
  region2='[4, 370, 690, 670]' # no keypose no sketch
  #region2='[231, 360, 791, 640]'
  # region3_prompt='[a <thanos1> <thanos2>, purple armor, near the castle, 4K, high quality, high resolution, best quality]'
  # region3_prompt='[a <thanos1> <thanos2>, purple armor, near the mountain, 4K, high quality, high resolution, best quality]'
  region3_prompt='[a <dog3> <dog4>, in the right]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[4, 720, 512, 1000]' # no keypose no sketch
  #region3='[267, 700, 829, 895]'
  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"
  # prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"
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
  seed_list="[14, 16, 17, 18, 22, 24, 30, 32, 34, 36]"
  python3 inf3.py \
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
 