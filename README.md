TODO: allow no-prompt image-content-detection and video creation
TODO: chain prompts, and each video starts with last frame from previous video


# If you are looking for a guide for how to run all of this:

If you have a graphics card with enough VRAM to run it, the chinese text-to-video model is surprisingly simple to get running.
I think you need at least 16gb vram, but it appears to be higher for more frames generated and lower for fewer.
https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis

This assumes windows platform, but its very similar in mac and linux (just smaller market share = less tutorials and updates)

To do so, you will need:

"conda" - miniconda3, look up how you can use conda to manage software dependencies

"git" - git to git clone the software onto your machine, including the code and the trained model

"cuda" - assuming you have nvidia GPU, you have to install cuda on your machine. Install cuda 11.7

Understanding of how to run commands in a command prompt window - I use anaconda powershell prompt

Python code understanding - you need to at least be able to open a text file and edit some code lines. Simple stuff though


Everything below was quick, took me about 15mins from start to finish on fast internet.

Install miniconda, install git for windows

open up miniconda command prompt (it should be a new shortcut in start menu called "anaconda powershell prompt", and navigate to where you want to put this code (ie "cd /videogen" etc).

"git clone https://github.com/alexvw/sd-video.git" will download the code here

"git clone https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis.git" will download the model and config.

Go into windows explorer, and move everything from the model into a folder named /model inside the /sd-video folder

You are now ready to start preparing your conda environment for this

"conda create -n videogen python=3.10.9" - Create a new miniconda environment called "videogen", which we will install all the required dependencies to

"conda activate videogen" - activate that environment

"conda install cuda -c nvidia/label/cuda-11.7.0 -c nvidia/label/cuda-11.7.1" - install cuda dependencies in python. Press y if prompted

make sure you are in the "sd-video" folder, and run "pip install -r requirements.txt" - this will install dependencies specific to this code

"conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia" make sure you have all the torch dependencies

navigate to the root directory, and find a file named "generate.py"

Within this file, you will see a set of parameters. These are used to change the input file, prompt, tweak performance and resolution, etc.

now, on your command line in, type "python .\generate.py" - 
Look inside the /output folder - there you will see 16 frames of a video.
Some tips - the configuration.json file in /model contains some parameters you can tweak. Upping '"max_frames": 16,' uses more vram, reducing it lowers vram usage.
