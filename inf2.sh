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
  fused_model="experiments/prompt2/combined_model_base"
  expdir="flower_1+vase_chilloutmix"

  #keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose_2x/harry_hermione_thanos_2x.png'
  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition=''
  #sketch_condition='datasets/validation_spatial_condition/multi-objects/flower_vase6.jpg'
  sketch_adaptor_weight=1.0
  # A <flower_1> in a <vase>.
  # context_prompt='three people near the castle, 4K, high quality, high resolution, best quality'
  # context_prompt='a <potter1> <potter2>, a <hermione1> <hermione2> and a <thanos1> <thanos2> near the castle, 4K, high quality, high resolution, best quality'
 # context_prompt='three people near the castle, 4K, high quality, high resolution, best quality'
  # context_prompt='a <potter1> <potter2>, a <hermione1> <hermione2> and a <thanos1> <thanos2> near the castle, 4K, high quality, high resolution, best quality'
  # context_prompt='A cat on the right and a dog on the left'
  context_prompt='a <flower1> <flower2> in a <vase1> <vase2>'
  # context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,fewer digits, cropped, worst quality, low quality'
  context_neg_prompt='extra digit, fewer digits, cropped, worst quality, low quality'

  # region1_prompt='[a <potter1> <potter2>, in Hogwarts uniform, holding hands, near the castle, 4K, high quality, high resolution, best quality]'
  region1_prompt='[a <flower1> <flower2> in a vase]'
  region1_neg_prompt="[${context_neg_prompt}]"
  # region1='[4, 6, 1024, 490]'
  # in the image size 512x512
  # region1='[2, 1302, 1024, 1992]'
  # region1='[4, 800, 512, 1992]'
  region1='[800, 500, 1024,1000 ]' # no pose no sketch
  #region1='[740, 412, 944, 643]' # sketch for flower_vase.jpg  and flower_vase5.jpg
  #region1='[515, 270, 950, 820]' # sketch for flower_vase3.jpg
  #region1='[505, 331, 876, 731]' # sketch FOR flower_vase4.jpg
 # region1='[636, 371, 924, 680]'


  region2_prompt='[<vase1> <vase2>, on the table]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[4, 500, 700, 1000]' # no pose no sketch
  #region2='[472, 477, 745, 559]' # sketch for flower_vase.jpg and flower_vase5.jpg
  #region2='[40,  420, 490, 664]' # sketch for flower_vase3.jpg
  #region2='[50, 423, 528, 581]' # sketch FOR flower_vase4.jpg
  # region2='[2, 1302, 1024, 1992]'
  #region2='[284, 443, 664, 570]'
  # region3_prompt='[a <thanos1> <thanos2>, purple armor, near the castle, 4K, high quality, high resolution, best quality]'
  # region3_prompt='[a <thanos1> <thanos2>, purple armor, near the mountain, 4K, high quality, high resolution, best quality]'
  # region3_neg_prompt="[${context_neg_prompt}]"
  # region3='[2, 1302, 1024, 1992]'

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
  seed_list='[14, 18, 26, 29, 30, 42, 54, 76, 93, 94, 95]' # flower_vase

  python3 inf2.py \
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
 