import argparse
import hashlib
import json
import os.path

import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers.models import T2IAdapter
from PIL import Image

from mixofshow.pipelines.pipeline_regionally_t2iadapter import RegionallyT2IAdapterPipeline


def sample_image(pipe,
    input_prompt,
    input_neg_prompt=None,
    generator=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    sketch_adaptor_weight=1.0,
    region_sketch_adaptor_weight='',
    keypose_adaptor_weight=1.0,
    region_keypose_adaptor_weight='',
    **extra_kargs
):

    keypose_condition = extra_kargs.pop('keypose_condition')
    if keypose_condition is not None:
        keypose_adapter_input = [keypose_condition] * len(input_prompt)
    else:
        keypose_adapter_input = None

    sketch_condition = extra_kargs.pop('sketch_condition')
    if sketch_condition is not None:
        sketch_adapter_input = [sketch_condition] * len(input_prompt)
    else:
        sketch_adapter_input = None

    images = pipe(
        prompt=input_prompt,
        negative_prompt=input_neg_prompt,
        keypose_adapter_input=keypose_adapter_input,
        keypose_adaptor_weight=keypose_adaptor_weight,
        region_keypose_adaptor_weight=region_keypose_adaptor_weight,
        sketch_adapter_input=sketch_adapter_input,
        sketch_adaptor_weight=sketch_adaptor_weight,
        region_sketch_adaptor_weight=region_sketch_adaptor_weight,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        **extra_kargs).images
    return images


def build_model(pretrained_model, device):
    pipe = RegionallyT2IAdapterPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16).to(device)
    assert os.path.exists(os.path.join(pretrained_model, 'new_concept_cfg.json'))
    with open(os.path.join(pretrained_model, 'new_concept_cfg.json'), 'r') as json_file:
        new_concept_cfg = json.load(json_file)
    pipe.set_new_concept_cfg(new_concept_cfg)
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_model, subfolder='scheduler')
    pipe.keypose_adapter = T2IAdapter.from_pretrained('TencentARC/t2iadapter_openpose_sd14v1', torch_dtype=torch.float16).to(device)
    pipe.sketch_adapter = T2IAdapter.from_pretrained('TencentARC/t2iadapter_sketch_sd14v1', torch_dtype=torch.float16).to(device)
    return pipe


def prepare_text(prompt, region_prompts, height, width):
    '''
    Args:
        prompt_entity: [subject1]-*-[attribute1]-*-[Location1]|[subject2]-*-[attribute2]-*-[Location2]|[global text]
    Returns:
        full_prompt: subject1, attribute1 and subject2, attribute2, global text
        context_prompt: subject1 and subject2, global text
        entity_collection: [(subject1, attribute1), Location1]
    '''
    region_collection = []

    regions = region_prompts.split('|')

    for region in regions:
        if region == '':
            break
        prompt_region, neg_prompt_region, pos = region.split('-*-')
        prompt_region = prompt_region.replace('[', '').replace(']', '')
        neg_prompt_region = neg_prompt_region.replace('[', '').replace(']', '')
        pos = eval(pos)
        if len(pos) == 0:
            pos = [0, 0, 1, 1]
        else:
            pos[0], pos[2] = pos[0] / height, pos[2] / height
            pos[1], pos[3] = pos[1] / width, pos[3] / width
        # pos = [0, 0, 1, 1]
        region_collection.append((prompt_region, neg_prompt_region, pos))
    return (prompt, region_collection)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_model', default='experiments/composed_edlora/anythingv4/hina+kario+tezuka+mitsuha+son_anythingv4/combined_model_base', type=str)
    parser.add_argument('--sketch_condition', default=None, type=str)
    parser.add_argument('--sketch_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_sketch_adaptor_weight', default='', type=str)
    parser.add_argument('--keypose_condition', default=None, type=str)
    parser.add_argument('--keypose_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_keypose_adaptor_weight', default='', type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--prompt', default='photo of a toy', type=str)
    parser.add_argument('--negative_prompt', default='', type=str)
    parser.add_argument('--prompt_rewrite', default='', type=str)
    parser.add_argument('--seed', default=16141, type=int)
    parser.add_argument('--suffix', default='', type=str)
    return parser.parse_args()

import time
if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = build_model(args.pretrained_model, device)

    if args.sketch_condition is not None and os.path.exists(args.sketch_condition):
        sketch_condition = Image.open(args.sketch_condition).convert('L')
        width_sketch, height_sketch = sketch_condition.size
        print('use sketch condition')
    else:
        sketch_condition, width_sketch, height_sketch = None, 0, 0
        print('skip sketch condition')

    if args.keypose_condition is not None and os.path.exists(args.keypose_condition):
        keypose_condition = Image.open(args.keypose_condition).convert('RGB')
        width_pose, height_pose = keypose_condition.size
        print('use pose condition')
    else:
        keypose_condition, width_pose, height_pose = None, 0, 0
        print('skip pose condition')

    if width_sketch != 0 and width_pose != 0:
        assert width_sketch == width_pose and height_sketch == height_pose, 'conditions should be same size'
    # width, height = max(width_pose, width_sketch), max(height_pose, height_sketch)
    width, height = 1024, 1024
    kwargs = {
        'sketch_condition': sketch_condition,
        'keypose_condition': keypose_condition,
        'height': height,
        'width': width,
    }

    

    image_list = []
    # 10 seeds
    # dog6+cat2
    # seed_list = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    # flower+vase(40,47, 76, 93, 94)
    # seed_list = [14, 18, 26, 29, 30, 42, 54, 76, 93, 94, 95]
    # seed_list = [101, 102, 103, 104, 105]
    # seed_list = [106, 107, 108, 109, 110]
    # seed_list = [111, 112, 113, 114, 115]
    # seed_list = [116, 117, 118, 119, 120]
    # seed_list = [i for i in range(101, 201)]
    # dog+petcat+dog6(32,41, 43)
    # seed_list = [14, 15, 16, 17, 18, 32,39, 43, 45, 47]
    # seed_list = [48, 49, 50, 51, 52]
    # seed_list = [53, 54, 55, 56, 57]
    # seed_list = [58, 59, 60, 61, 62]
    # seed_list = [63, 64, 65, 66, 67]

    # style(16,20,22,23, 24, 31, 40)
    # seed_list = [14]
    # seed_list = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    seed_list = [14, 16, 17, 18, 22, 24, 30, 32, 34, 36]
    
    # seed_list = [24, 25, 26, 27, 28, 29, 30, 31]
    # seed_list = [32, 33, 34, 35, 36, 37, 38, 39]
    # seed_list = [40, 41, 42, 43, 44, 45, 46, 47]
    for i in range(10):
        prompts = [args.prompt]
        prompts_rewrite = [args.prompt_rewrite]
        input_prompt = [prepare_text(p, p_w, height, width) for p, p_w in zip(prompts, prompts_rewrite)]
        save_prompt = input_prompt[0][0]
        image = sample_image(
            pipe,
            input_prompt=input_prompt,
            input_neg_prompt=[args.negative_prompt] * len(input_prompt),
            generator=torch.Generator(device).manual_seed(seed_list[i]),
            sketch_adaptor_weight=args.sketch_adaptor_weight,
            region_sketch_adaptor_weight=args.region_sketch_adaptor_weight,
            keypose_adaptor_weight=args.keypose_adaptor_weight,
            region_keypose_adaptor_weight=args.region_keypose_adaptor_weight,
            **kwargs)
        # if i == 0:
        #     print("length of image: ", len(image))
        #     print("image", image)
        image_list.append(image[0])

    # print(f'save to: {args.save_dir}')

    configs = [
        f'pretrained_model: {args.pretrained_model}\n',
        f'context_prompt: {args.prompt}\n', f'neg_context_prompt: {args.negative_prompt}\n',
        f'sketch_condition: {args.sketch_condition}\n', f'sketch_adaptor_weight: {args.sketch_adaptor_weight}\n',
        f'region_sketch_adaptor_weight: {args.region_sketch_adaptor_weight}\n',
        f'keypose_condition: {args.keypose_condition}\n', f'keypose_adaptor_weight: {args.keypose_adaptor_weight}\n',
        f'region_keypose_adaptor_weight: {args.region_keypose_adaptor_weight}\n', f'random seed: {args.seed}\n',
        f'prompt_rewrite: {args.prompt_rewrite}\n'
    ]
    hash_code = hashlib.sha256(''.join(configs).encode('utf-8')).hexdigest()[:8]
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    save_prompt = save_prompt.replace(' ', '_')
    # save_name = f'{save_prompt}.png'
    save_name = f'prompt_3.png'
    # save_dir = os.path.join(args.save_dir, f'{current_time}')
    save_dir = "./sample_submission/2"
    save_path = os.path.join(save_dir, save_name)
    save_config_path = os.path.join(save_dir, save_name.replace('.png', '.txt'))

    os.makedirs(save_dir, exist_ok=True)
    # resize image to 512x512
    # image[0] = image[0].resize((512, 512), Image.LANCZOS)
    # image[0].save(os.path.join(save_dir, save_name))
    for i in range(10):
        image_list[i] = image_list[i].resize((512, 512), Image.LANCZOS)
        image_list[i].save(os.path.join(save_dir, f'{save_name.replace(".png", f"_{i}_seed_{seed_list[i]}.png")}'))

    # with open(save_config_path, 'w') as fw:
    #     fw.writelines(configs)
