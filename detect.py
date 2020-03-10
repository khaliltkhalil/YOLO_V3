from darknet import *
from helper import *
import torch
import cv2
import matplotlib.pyplot as plt

def load_network(cnfg_file, weight_file):
    num_classes = 80
    names = get_names("data/coco.names")

    print("Loading network.....")
    model = Darknet(cnfg_file)
    model.load_weights(weight_file)
    print("Network loaded Successfully")

    return model, names



def detect(img_file, model, names, threshold=0.9, nms_thr=0.6):

    inp_dim = int(model.net_info["height"])
    num_classes = 80
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_new, padding_st_indx, scale_factor = resize_image(img, inp_dim)
    #plt.imshow(img_new)
    st_index_h = padding_st_indx[0]
    st_index_w = padding_st_indx[1]

    # convert from numpy to tensore with B X C X H X W
    img_new = img_new.transpose((2, 0, 1))  # C X H X W
    img_new = img_new / 255
    img_new = torch.from_numpy(img_new).float()
    img_new = img_new.unsqueeze(0)

    input = img_new

    prediction = model(input, False)
    detection = write_results(prediction, threshold, num_classes, nms_thr)


    detection[:,[1,3]] -= st_index_w
    detection[:,[2,4]] -= st_index_h
    detection[:,1:5] /= scale_factor

    detection[:,[1,3]] = torch.clamp(detection[:,[1,3]],0,img.shape[1])
    detection[:,[2,4]] = torch.clamp(detection[:,[2,4]], 0, img.shape[0])

    boxes = detection[:,1:]
    boxes = boxes.data.numpy()
    img_detected = draw_boxes(img,boxes,names)
    fig = plt.figure(figsize=(10, 10))
    axes = fig.add_axes([0, 0, 1, 1])
    axes.imshow(img_detected)







