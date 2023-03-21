from sd_video import SDVideo, save_vid, save_webm
model = SDVideo('model', 'cuda')
x = model('turtle boy standing there holding stick')
#save_vid(x, 'output')
save_webm(x, 'output', 'output.webm', 12) # 0001.png ... 00NN.png