# DLCV Final Project ( Multiple Concept Personalization )



# Inference:
# Environment setup
```shell script=
git clone https://github.com/DLCV-Fall-2024/DLCV-Fall-2024-Final-2-chillguy.git
cd DLCV-Fall-2024-Final-2-chillguy
mkdir experiments && cd experiments
mkdir pretrained_models && cd pretrained_models
# Diffusers-version ChilloutMix
git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git

cd ../../
conda create -n dlcvteam10 python=3.9
conda activate dlcvteam10
# For cuda=12.2
pip3 install torch torchvision torchaudio
pip install xformers
pip install -r requirements.txt
```
# Download Dataset

```shell script=
bash download_ckpt.sh
```
# Run infernece
* Run all
```shell script=
 bash run_all.sh
```
* Run each prompt individually
```shell script=
bash inf1.sh
bash inf2.sh
bash inf3.sh
bash inf4.sh
```
* The generated images will be saved in the `sample_submission/<prompt_id>` folder.
* For example, the generated images for prompt 0 will be saved in the `sample_submission/0` folder.
 
# Training
* Inferece can just run without training. Just follow the above steps
# Download Dataset
```shell script=
bash download_train.sh
```
# Run single concept training
* concept list
    * cat2.yml
    * dog6.yml
    * flower_1.yml
    * dog.yml
    * pet_cat1.yml
    * vase.yml
    * watercolor.yml
    * wearable_glasses.yml
```shell script=
bash train.sh  <concept.yaml>
```
* For example, to train the model for concept `cat2.yml`, run the following command:
```shell script=
bash train.sh cat2.yml
```
* The trained model will be saved in the `experiments/<concept>/models` folder.
* For example, the trained model for concept `cat2.yml` will be saved in the `experiments/cat2/models/edlora_model-latest.pth` folder.
* `edlora_model-latest.pth` is the lora weights checkpoint of the single concept which will be used for fusion training.


# Fusion Training
* After training the single concept models, we can find the checkpoint files in the `experiments/<concept>/models` folder.
* We can use the following command to train the fusion model. 
* Therefore, you need to train the model for each concept first and put the checkpoints in the `experiments/<concept>/models` folder.
* prompt list
    * 1: cat2, dog6
    * 2: flower_1, vase
    * 3: dog, pet_cat1, dog6
    * 4: cat2, dog6, watercolor, wearable_glasses
```shell script=
bash fuse.sh <prompt_id>
```
* For example, to train the model for prompt 1, run the following command:
```shell script=
bash fuse.sh 1
```
* The fused model will be saved in the `experiments/prompt<prompt_id>/` folder.
* For example, the fused model for prompt 1 will be saved in the `experiments/prompt1/` folder.
* Then, you can follow the instructions above to run the inference script to generate the images for the prompt.
* For example, to generate the images for prompt 1, run the following command:
```shell script=
bash inf1.sh
```

# Usage
To start working on this final project, you should clone this repository into your local machine by the following command:

    git clone https://github.com/DLCV-Fall-2024/DLCV-Fall-2024-Final-2-<team name>.git
  
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1eeXx_dL0OgkDn9_lhXnimTHrE6OYvAiiVOBwo2CTVOQ/edit?usp=sharing) to view the slides of Final Project - Multiple Concept Personalization. **The introduction video for final project can be accessed in the slides.**

# Submission Rules
### Deadline
113/12/26 (Thur.) 23:59 (GMT+8)
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under `[Final challenge 2] Discussion` section in NTU Cool Discussion
