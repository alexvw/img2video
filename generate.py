from sd_video import SDVideo, save_webm

model_path = 'model' #where to find the damo/text-to-video-synthesis model
device = 'cuda' #cuda, cpu not tested
text = 'sea turtle swimming, right to left, blue water, underwater, ocean'
text_neg = 'watermark'
initial_alpha = 0.52 #how hard to influence the first frame. 0.23 is a pretty good default value
ratio = 0.8 #how much to influence each frame. is multiplied against previous alpha, 0.8 works alright
#TODO this ratio is useful because you can tell when the noise goes from image-influenced to totally random and natural video. Right now, that means you can tell when the color channels return to normal, as all the weird red and purple leaves and the video becomes normal colored

model = SDVideo(model_path, device)

#model(text, text_neg, 30, 0.3, 0.7, 'turtle.png', 'output/turtle-test.webm', 10)
#self, multiline_prompt: str, image_path: str, max_frames: int = 16, initial_alpha: float = 0.23, ratio: float = 0.8, output_file_path: str = "output.webm", fps: int = 16


multiline_prompt = """darth maul appears and shows his red lightsaber, desert, tatooine
darth maul jump attack, desert, jump attack
luke skywalker battle lightsaber, darth maul fight, desert, tatooine
darth maul lightsaber red attack
darth maul wins, walks away, skywalker dies, red lightsaber kill, blue lightsaber death, desert
star wars spaceship, imperial spaceship flies away
darth star wars spaceship, in the atmosphere, flying in space, stars
star wars spaceship, flies away, warp speed"""
model.process_multiline_prompt(multiline_prompt, image_path="input.png", max_frames=36, initial_alpha=0.23, ratio=0.8, output_file_path="output/starwars-space.webm", fps=8)
