'''
yolov3 anchor (k-means clustering) using DarkNet
'''
pascal_voc = {
    "anchors" : [[[328, 193], [218, 328], [361, 362]], # 13 X 13
                [[72, 195], [170, 178], [122, 296]], # 26 X 26
                [[26, 38], [46, 97], [118, 92]]], # 52 X 52
    "classes" : 20,
}

coco = {
    "anchors" : [[[116, 90], [156, 198], [373, 326]],
                [[30, 61], [62, 45], [59, 119]],
                [[10, 13], [16, 30], [33, 23]]],
    "classes" : 80,
}