import cv2
import numpy as np

import time
import torch
import torch.nn as nn

def acc_check(net, device, test_set_loader, epoch, save_path):
    net.eval()
    net.is_training = False
    with torch.no_grad():
        total_time = 0
        for i, data in enumerate(test_set_loader, 0):
            start = time.time()
            images, labels, name = data

            images = images.to(device)

            outputs = net(images)
            process_time = time.time() - start
            print("Process Time in a Validation Image {}".format(process_time))
            total_time += process_time

            outputs = (outputs[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

            save_name = save_path + "{0}_{1}.png".format(epoch, name[0].split("\\")[-1].split(".")[0])
            cv2.imwrite(save_name, outputs.astype(np.uint8))

        print("Average time of total process {}".format(total_time / test_set_loader.__len__()))
    net.is_training = True
    net.train()

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr

def value_tracker(vis, num, value, value_plot):
    ''' num, loss_value, are Tensor '''
    vis.line(X=num,
             Y=value,
             win=value_plot,
             update='append')

def decode_segmap(output, nc=3):
    output = torch.argmax(output, dim=0).detach().cpu().numpy()

    r = np.zeros_like(output).astype(np.uint8)
    g = np.zeros_like(output).astype(np.uint8)
    b = np.zeros_like(output).astype(np.uint8)

    label_colors = np.array([(255, 0, 255), (0, 0, 255), (0, 0, 0)])  # road, non-road, background

    for l in range(0, nc):
        idx = output == l
        r[idx] = label_colors[l, 2]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 0]
    rgb = np.stack([b, g, r], axis=2)

    return rgb