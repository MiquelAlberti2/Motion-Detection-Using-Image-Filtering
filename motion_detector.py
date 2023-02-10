def threshold(value):
    if value > .3: # Change threshold value here
        return 1
    else:
        return 0

images = [] # array of 3 images, each 320x240
for i in range(1,1069):
    with open('Office/image01_'+str(i).zfill(4)+'.jpg', 'rb') as f:
        images.append(f.read())
    with open('Office/image01_'+str(i+1).zfill(4)+'.jpg', 'rb') as f:
        images.append(f.read())
    with open('Office/image01_'+str(i+2).zfill(4)+'.jpg', 'rb') as f:
        images.append(f.read())
    mask = [] # new image of size 320x240
    for row in range(320):
        mask.append([])
        for col in range(240):
            mask[row].append(threshold(.5*(-1*images[0][row][col] + 1*images[2][row][col])))
    maskedImage = [] # new image of size 320x240
    for row in range(320):
        maskedImage.append([])
        for col in range(240):
            maskedImage[row].append(images[1][row][col]*mask[row][col])


