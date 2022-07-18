random.seed(0)
class_colors = [[0,0,0],[120,120,120]]
NCLASSES = 2
HEIGHT = 3200
WIDTH = 3200

h5_path = r"../input/h5-10-20220402/1Compact_highrise/"
for file_name in (os.listdir(h5_path)):
    model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    model.load_weights(h5_path+"/"+file_name)
    imgs = os.listdir("../input/xiamen2003/siming2003/")
    for jpg in imgs:
        img = Image.open("../input/xiamen2003/siming2003/"+jpg)
        old_img = copy.deepcopy(img)
        orininal_h = np.array(img).shape[0]
        orininal_w = np.array(img).shape[1]
        img = img.resize((WIDTH,HEIGHT))
        img = np.array(img)
        img = img/255
        img = img.reshape(-1,HEIGHT,WIDTH,3)
        pr = model.predict(img)[0]
        pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2),NCLASSES)).argmax(axis=-1)
        seg_img = np.zeros((int(HEIGHT/2), int(WIDTH/2),3))
        colors = class_colors
        for c in range(NCLASSES):
            seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
        seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        image = Image.blend(old_img,seg_img,1)
        image.save("./siming_1/"+file_name.split('.')[0]+'_'+jpg) 