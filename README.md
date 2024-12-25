# DLCV Final Project ( Multiple Concept Personalization )



# Inference:
# Environment setup
```shell script=
git clone https://github.com/DLCV-Fall-2024/DLCV-Fall-2024-Final-2-chillguy.git
mkdir experiments && cd experiments
mkdir pretrained_models && cd pretrained_models
# Diffusers-version ChilloutMix
git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git

cd ../../
conda create -n dlcvteam10 python=3.9
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

```shell script=
bash train.sh <Path to gt image folder> <Path to annot file>
bash inference.sh <Path to gt image folder> <Path to annot file> <Path to output image folder>
```
# Training

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
