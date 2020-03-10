import torch
import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = False):
    #print(prediction.size())
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    #print(grid_size)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    #anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)


    prediction[:,:,0:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)

    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    prediction[:,:,5:] = torch.sigmoid(prediction[:,:,5:])

    prediction[:,:,0:2] *= stride

    return prediction

def unique(tensor):
    tensor_np = tensor.detach().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    return unique_tensor

def bbox_iou(box, boxes):

    b1_x1, b1_y1, b1_x2, b1_y2 = box[0], box[1], box[2], box[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0 ) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):

    # mask all boxes with object score less than confidence to zeros
    mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * mask


    # transform  (center x, center y, width, hieht)--
    # to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y).
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = prediction[:,:,0] - prediction[:,:,2]/2
    box_corner[:,:,1] = prediction[:,:,1] - prediction[:,:,3]/2
    box_corner[:,:,2] = prediction[:,:,0] + prediction[:,:,2]/2
    box_corner[:,:,3] = prediction[:,:,1] + prediction[:,:,3]/2
    prediction[:,:,0:4] = box_corner[:,:,0:4]

    batch_size = prediction.size(0)

    write = False
    # loop over each image in the batch
    for i in range(batch_size):
        img_pred = prediction[i]

        max_conf, max_conf_indx = torch.max(img_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().view(-1,1)
        max_conf_indx = max_conf_indx.float().view(-1,1)
        img_pred = torch.cat((img_pred[:,:5],max_conf,max_conf_indx), 1)

        nonzero_ind = torch.nonzero(img_pred[:,4]).squeeze()
        try:
            img_pred_ = img_pred[nonzero_ind,:].view(-1,7)

        except:
            continue

        # get the detected classes not repeated
        img_classes = unique(img_pred_[:,-1])
        #print(img_classes)
        for cls in img_classes:
            img_pred_class = img_pred_[img_pred_[:,-1] == cls]

            # sort the class detection with the highest prob on top
            indx_sort = torch.sort(img_pred_class[:,4],descending= True)[1]
            img_pred_class = img_pred_class[indx_sort]
            #print(img_pred_class.size())
            #print(img_pred_class)
            num_detect = img_pred_class.size(0)

            for k in range(num_detect):
                try:
                    iou = bbox_iou(img_pred_class[k], img_pred_class[k+1:])
                    #print(iou)
                except ValueError:
                    break
                except IndexError:
                    break
                img_pred_class_rest = img_pred_class[k+1:]
                img_pred_class = torch.cat((img_pred_class[:k+1], img_pred_class_rest[iou < nms_conf]), 0)

            batch_ind = img_pred_class.new(img_pred_class.size(0), 1).fill_(i)
            seq = (batch_ind, img_pred_class)
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out),0)

        try:
            return output
        except:
            return 0

def get_names(names_file):
    file = open(names_file)
    names = file.read().split("\n")[:-1]
    return names

def resize_image(img, inp_dim):
    ''' resize image while keeping aspect ratio unchanged using padding'''
    img_w = img.shape[1]
    img_h = img.shape[0]
    aspect_ratio = img_h / img_w
    if aspect_ratio < 1:
        new_w = inp_dim
        new_h = int(new_w * aspect_ratio)
    else:
        new_h = inp_dim
        new_w = int(new_h / aspect_ratio)
    scale_factor = new_h/img_h
    new_image = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
    padded_img = np.full((inp_dim, inp_dim, 3), 128)

    padding_size_h = inp_dim - new_h
    padding_size_w = inp_dim - new_w
    start_indx_h = int(padding_size_h / 2)
    start_indx_w = int(padding_size_w / 2)

    padded_img[start_indx_h:start_indx_h + new_h, start_indx_w:start_indx_w + new_w,:] = new_image
    padded_img = padded_img.astype(np.uint8)
    return padded_img, (start_indx_h, start_indx_w), scale_factor

def prep_image(img, inp_dim):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = resize_image(img, inp_dim)
    img = img.transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255).unsqueeze(0)
    return img

def draw_boxes(img, boxes, classes):
    for i in range(boxes.shape[0]):
        pt1 = tuple(boxes[i,[0,1]])
        pt2 = tuple(boxes[i,[2,3]])
        label = classes[int(boxes[i,-1])]
        cv2.rectangle(img, pt1,pt2,[255,0,0],3)
        t_size = cv2.getTextSize(label,cv2.FONT_HERSHEY_PLAIN,1,1)[0]
        pt3 = (int(pt1[0] + t_size[0]+3), int(pt1[1] + t_size[1]+3))
        cv2.rectangle(img,pt1,pt3,[255,0,0],-1)
        cv2.putText(img,label,(int(pt1[0]),int(pt1[1]+t_size[1])),cv2.FONT_HERSHEY_PLAIN,1,[0,0,255],2)
    return img
















