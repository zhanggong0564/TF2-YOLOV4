#coding:utf-8
import os
import json
import cv2


def json_visualization(json_path,img_path,list_file):
    classes = {'car': 0, 'traffic light': 1, 'pedestrian': 2}
    with open(json_path, 'r') as json_file:
        #print(json_file)
        for line in json_file:
            data = json.loads(line)
            for d in data:
                image_path = os.path.join(img_path,d['name'])
                # image = cv2.imread(image_path)
                wirte_path = False
                for ob in d['labels']:
                    li = []
                    if ob['category'] in ['car','traffic light','pedestrian']:
                        if not wirte_path:
                            list_file.write(image_path)
                            li.append(image_path)
                            wirte_path = True
                        b = (int(ob['box2d']['x1']),int(ob['box2d']['y1']),int(ob['box2d']['x2']),int(ob['box2d']['y2']))
                        # print((int(ob['box2d']['x2'])-int(ob['box2d']['x1']))*(int(ob['box2d']['y2'])-int(ob['box2d']['y1'])))
                        list_file.write(" " + ",".join([str(a) for a in b])+','+str(classes[ob['category']]))
                        # cv2.rectangle(image,pt1=(int(ob['box2d']['x1']),int(ob['box2d']['y1'])),
                        #               pt2=(int(ob['box2d']['x2']),int(ob['box2d']['y2'])),color=(0,255,28))
                        # cv2.putText(image,ob['category'],(int(ob['box2d']['x1']),int(ob['box2d']['y1'])),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0))
                if wirte_path:
                    list_file.write('\n')
                # cv2.imshow('test',image)
                # cv2.waitKey()
                # cv2.destroyAllWindows()


def image_json_visualization(img_dir,json_dir,list_file):
    '''
    :param img_dir: 图片所在文件夹
    :param json_dir: json文件所在文件夹
    :param save_dir: 可视化后图片保存路径
    :return:
    '''
    marksave_dir=os.path.join(save_dir,'mark1')
    if not os.path.exists(marksave_dir):
        os.makedirs(marksave_dir)

    json_visualization(json_dir,img_dir,list_file)
if __name__ == '__main__':
    img_dir = r'H:\data\auto_dirve\Images\100k\val'
    json_dir = r'H:\data\auto_dirve\bdd100k_labels_detection20\bdd100k\labels\detection20\det_v2_val_release.json'
    save_dir = r'./json_visual'
    list_file = open('train_val.txt', 'w')
    image_json_visualization(img_dir,json_dir,list_file)
    list_file.close()

