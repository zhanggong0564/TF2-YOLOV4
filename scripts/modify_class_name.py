import os
input_dir='/media/zhanggong/Elements/yolov4/TF2-YOLOV4/VOCdevkit/VOC2007/Annotations'
shu=0
new_name1='green apple'
import xml.etree.ElementTree as ET
save = False
for filename in os.listdir(input_dir):
    print(filename)
    if 'xml' in filename:
        file_path = os.path.join(input_dir, filename)
        dom = ET.parse(file_path)
        root = dom.getroot()
        for obj in root.iter('object'):  # 获取object节点中的name子节点
            if obj.find('name').text== 'green_apple':
                obj.find('name').text=new_name1
                shu=shu+1
                save = True
                #print("change %s to %s." % (yuan_name, new_name1))
            # elif obj.find('name').text== 'l':
            #     obj.find('name').text = new_name2
            #     shu = shu + 1
            # elif obj.find('name').text== 'lf':
            #     obj.find('name').text= new_name3
            #     shu = shu + 1
            # elif obj.find('name').text == 'ic':
            #     obj.find('name').text= new_name4
            #     shu = shu + 1
            # elif obj.find('name').text== 'bt':
            #     obj.find('name').text= new_name5
            #     shu = shu + 1
            # elif obj.find('name').text== 'ipr':
            #     obj.find('name').text= new_name6
            #     shu = shu + 1
            # elif obj.find('name').text== 'c':
            #     obj.find('name').text= new_name7
            #     shu = shu + 1
      # 保存到指定文件
        if save:
            dom.write(file_path, xml_declaration=True)
print("有%d个文件被成功修改。" % shu)