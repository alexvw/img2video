from sd_video import SDVideo, save_vid, save_webm
model = SDVideo('model', 'cuda')
x = model('camera pan of new york skyline')
#save_vid(x, 'output')
save_webm(x, 'output', 'output.webm', 12) # 0001.png ... 00NN.png