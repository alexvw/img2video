### Usage:
```
model = SDVideo(model_path, device)

multiline_prompt = """darth maul appears and shows his red lightsaber, desert, tatooine
darth maul jump attack, red lightsaber, desert, jump attack
star wars spaceship, imperial spaceship flies away
star wars spaceship, flies away, space, stars, warp speed"""
model.process_multiline_prompt(multiline_prompt, image_path="input.png", max_frames=10, initial_alpha=0.23, ratio=0.8, output_file_path="output/starwars-space.webm", fps=8)
```

The code above will take in the input, break it down into lines, and generate a video for each line. Each video will be started with an img2video from the previous line's video, and at each step it will concatenate the webm frames onto the next, leaving a single output file at the end.

If you want a simple img2video, just pass in text as one line. 

### Parameters:

- multiline_prompt: the text to generate scenes from. Benefits from scene directions like camera pan left, zoom in, etc. Be descriptive visually
- image_path: an image to start the first video frame from. Using the current modelscope model, this is limited to a 32x32 dimension for the comparison so... the image will not be that similar. It will follow general color borders and layouts but don't expect img2img quality yet. (If you have ideas for how to improve this, try it! I would love a PR)
- max_frames: the max number of frames per each line/scene
- initial_alpha: how hard to weight the input image frame. Anything from 0.1-0.3 is usable, above that it will burn in and cause artifacts. 
- ratio: fade in weight for the initial image. Each successive frame weight will be multiplied with this ratio. For example- second frame will be 0.23*0.8 weight of the input image, and so on.
- output_file_path: where the output webm will save
- fps: the frames per second to render the images to webm. The model tends to generate fast motion, so slow fps works better.








# If you are looking for a guide for how to run all of this:

If you have a graphics card with enough VRAM to run it, the modelscope text2video (and this img2video fork) is surprisingly simple to get running.
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

Note: this repo is changing, and so do the dependencies. If you get an error saying something is missing, try "conda install <something>" and that will likely install it.
