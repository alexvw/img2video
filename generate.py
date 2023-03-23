from sd_video import SDVideo, save_vid, save_webm

model_path = 'model' #where to find the damo/text-to-video-synthesis model
device = 'cuda' #cuda, cpu not tested
text = 'camera zoom, green oak tree in field'
text_neg = 'watermark'
initial_alpha = 0.25 #how hard to influence the first frame. 0.23 is a pretty good default value
ratio = 0.8 #how much to influence each frame. is multiplied against previous alpha, 0.8 works alright
#TODO this ratio is useful because you can tell when the noise goes from image-influenced to totally random and natural video. Right now, that means you can tell when the color channels return to normal, as all the weird red and purple leaves and the video becomes normal colored


model = SDVideo(model_path, device)


image_path = 'tree.png' #path from the root to where the initial frame image is
x = model(text, text_neg, initial_alpha, ratio, image_path)
save_webm(x, 'output', 'tree.webm', 10) #x, folder, filename, fps


image_path = 'tree-1green.png' #path from the root to where the initial frame image is
x = model(text, text_neg, initial_alpha, ratio, image_path)
save_webm(x, 'output', 'tree-1green.webm', 10) #x, folder, filename, fps

image_path = 'tree-nogreen.png' #path from the root to where the initial frame image is
x = model(text, text_neg, initial_alpha, ratio, image_path)
save_webm(x, 'output', 'tree-nogreen.webm', 10) #x, folder, filename, fps

