from sd_video import SDVideo, save_webm

model_path = 'model' #where to find the damo/text-to-video-synthesis model
device = 'cuda' #cuda, cpu not tested
text = 'sea turtle, swimming past, underwater, ocean'
text_neg = 'watermark'
initial_alpha = 0.52 #how hard to influence the first frame. 0.23 is a pretty good default value
ratio = 0.8 #how much to influence each frame. is multiplied against previous alpha, 0.8 works alright
#TODO this ratio is useful because you can tell when the noise goes from image-influenced to totally random and natural video. Right now, that means you can tell when the color channels return to normal, as all the weird red and purple leaves and the video becomes normal colored


model = SDVideo(model_path, device)


image_path = 'turtle.png' #path from the root to where the initial frame image is
initial_alpha = 0.52 #how hard to influence the first frame. 0.23 is a pretty good default value
ratio = 0.8 #how much to influence each frame. is multiplied against previous alpha, 0.8 works alright


x = model(text, text_neg, 0.32, 0.5, image_path)
save_webm(x, 'output', 'turtle-0.32-0.6.webm', 20) #x, folder, filename, fps

x = model(text, text_neg, 0.27, 0.5, image_path)
save_webm(x, 'output', 'turtle-0.27-0.6.webm', 20) #x, folder, filename, fps

x = model(text, text_neg, 0.25, 0.5, image_path)
save_webm(x, 'output', 'turtle-0.25-0.6.webm', 20) #x, folder, filename, fps

x = model(text, text_neg, 0.23, 0.5, image_path)
save_webm(x, 'output', 'turtle-0.23-0.6.webm', 20) #x, folder, filename, fps

x = model(text, text_neg, 0.21, 0.5, image_path)
save_webm(x, 'output', 'turtle-0.21-0.6.webm', 20) #x, folder, filename, fps

