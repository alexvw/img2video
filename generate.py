from sd_video import SDVideo, save_vid, save_webm
model = SDVideo('model', 'cuda')
x = model('slow camera pan, studio ghibli style, anime, countryside, blue sky, farmhouse')
save_vid(x, 'output')
#save_webm(x, 'output', 'video.webm', 12) # 0001.png ... 00NN.png