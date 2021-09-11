import os
from txt2json_hsrc2016 import collect_unaug_dataset, convert
#img_dic = collect_unaug_dataset(os.path.join("trainset","DOTA_1024_1","labelTxt"))
#convert(img_dic, "trainset/DOTA_1024_1",os.path.join("trainset","DOTA_1024_1","train.json"))
img_dict = collect_unaug_dataset( os.path.join("trainset/HSRC2016_1","labelTxt"))
convert(img_dict,"trainset/HSRC2016_1",os.path.join("trainset/HSRC2016_1","train.json"))

