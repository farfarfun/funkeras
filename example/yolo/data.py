import os

import demjson
import pandas as pd

data_root = '/Users/liangtaoniu/tmp/dataset/livedata/Live_demo_20200117'
dir_video = data_root + '/video'
dir_image = data_root + '/image'
dir_video_text = data_root + '/video_text'
dir_image_text = data_root + '/image_text'

dir_video_annotation = data_root + '/video_annotation'
dir_image_annotation = data_root + '/image_annotation'

d = os.listdir(data_root)


def get_label(label):
    label_array = [
        ['短袖上衣', '短袖Top'],  # 0
        ['长袖上衣', '长袖Top'],
        ['短袖衬衫'],
        ['长袖衬衫'],
        ['背心上衣'],
        ['吊带上衣'],  # 5
        ['无袖上衣'],
        ['短外套'],
        ['短马甲'],
        ['长袖连衣裙'],
        ['短袖连衣裙'],  # 10
        ['无袖连衣裙'],
        ['长马甲'],
        ['长外套', '长款外套'],
        ['连体衣'],
        ['古风'],  # 15
        ['短裙'],
        ['中等半身裙', '中等半身裙（及膝）'],
        ['长半身裙'],
        ['短裤'],
        ['中裤'],  # 20
        ['长款', '长裤'],
        ['背带裤']]

    labels = {}
    for i, keys in enumerate(label_array):
        for key in keys:
            labels[key] = i

    if label in labels.keys():
        return labels[label]
    print(label)
    return 0


def image():
    dir_list = os.listdir(dir_image_annotation)
    data2 = []
    for dirs in dir_list:
        for file in os.listdir(os.path.join(dir_image_annotation, dirs)):
            file_path = os.path.join(dir_image_annotation, dirs, file)

            with open(file_path, 'r') as f:
                json = demjson.decode(f.read())

                annotations = json['annotations']
                if len(annotations) > 0:
                    for annotation in annotations:
                        annotation['label'] = get_label(annotation['label'])
                        for key in ['item_id', 'img_name']:
                            annotation[key] = json[key]

                        box = annotation['box']
                        annotation['train'] = '{},{},{},{},{}'.format(box[0], box[1], box[2], box[3],
                                                                      annotation['label'])
                        annotation['path'] = os.path.join(dir_image, annotation['item_id'], annotation['img_name'])
                        data2.append(annotation)
                        
    df = pd.DataFrame.from_dict(data2)
    print(df)
    df2 = df[['path', 'train']]
    df3 = df2.groupby(['path']).agg(lambda x: x.str.cat(sep=' '))
    df3 = df3.reset_index()
    path = '/Users/liangtaoniu/workspace/MyDiary/notechats/notekeras/example/yolo/data/dataset/yymnist_train.txt'
    df3.to_csv(path, index=None, quotechar=' ', sep=' ', header=None)


image()
