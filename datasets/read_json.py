# For Object Detection Task Only
# Read json label files, produced by LabelImg, into COCO-API dataset
# P.S. Up to now(11/8/2019), the latest version of LabelImg, will write
#       the image as base64 format into the json file, after labelling.
# P.S. One json file for one image and its all bbox(es).

import os
import glob
import json
import base64
import logging
import random
import argparse
args = None

# \brief  search into rootpath by recursive way
# \param  imglist: list
# \param  rootpath: string
# \param  recursivedepth: int
# \usage  used in write_as_cocolike
def recursive_loop_rootpath(imglist, rootpath, recursivedepth):
    if recursivedepth>0:
        for batch in os.listdir(rootpath):
            recursive_loop_rootpath(imglist, os.path.join(rootpath,batch),recursivedepth-1)
    else:
        imglist_ = []
        for img in glob.glob(os.path.join(rootpath,"*.json")):
            imglist_.append(img)
        imglist.extend(imglist_)
    return


# \brief  read json files and convert them as coco-like
# \param  args: parse_args
# \usage  used in __main__
def write_as_cocolike(args):
    id_img = 0; id_annotation = 0
    imglist = []
    dataset = {
        'categories': [],
        'images': [],
        'annotations': []}
    normal_bbox_id_list = []  # Collection of valid category id(s)
    error_bbox_id_dict = {}  # For those whose shape['label'] is not int,
                             # and new generated id(int) will be saved as {shape['label']: 8213,...},
                             # which the key will be treat as str, and the value(category id) will be randomly generated.


    # Get image list
    recursive_loop_rootpath(imglist, args.root_path, args.recursive_depth)


    for jsonfilename in imglist:
        # Read json file
        with open(jsonfilename, 'r') as jsonfile:
            try:
                jsonfileobj = json.load(jsonfile)
            except:
                logging.error("Json file {} is not dict-like.".format(jsonfilename))
                continue


        # Read & check the fields
        try:
            image_name = jsonfileobj["imagePath"]
        except:
            logging.error("Json file {} has not field 'imagePath'.".format(jsonfilename))
            continue
        try:
            image_data = jsonfileobj["imageData"]
        except:
            logging.error("Json file {} has not field 'imageData'.".format(jsonfilename))
            continue
        try:
            shapes_list = jsonfileobj["shapes"]
        except:
            logging.error("Json file {} has not field 'shapes'.".format(jsonfilename))
            continue


        # Write base64 codes as image onto disk
        image_path = os.path.join(args.output_path, image_name)
        try:
            with open(image_path,'wb') as fh:
                try:
                    data = base64.b64decode(image_data)
                except:
                    data = base64.decodebytes(image_data)
                fh.write(data)
        except:
            logging.error("Writing {} failed.".format(image_path))
            continue


        # Add an entry to "images" field
        dataset['images'].append({"file_name": image_path,
                                  "id": id_img,
                                  "width":jsonfileobj["imageWidth"],
                                  "height":jsonfileobj["imageHeight"]})


        # Add entry/entries to "annotations" field
        for shape in shapes_list:
            try:
                cls = int(shape['label'])
                normal_bbox_id_list.append(cls)
            except:  # type of shape['label'](category id) is invalid, and it will be treated as str(category name).
                if shape['label'] not in error_bbox_id_dict.keys():
                    newgenid = random.randint(90000,99999)
                    while newgenid in error_bbox_id_dict.values():
                        newgenid = random.randint(90000, 99999)
                    error_bbox_id_dict[shape['label']] = newgenid
                    cls = newgenid
                    logging.warning("Type of {} is invalid. Record in error_bbox_id_dict doesn't exist. "
                                    "Category id will be randomly generated. The bbox will be read as category id: {}".format(shape['label'], cls))
                else:
                    cls = error_bbox_id_dict[shape['label']]
                    logging.warning("Type of {} is invalid. Record in error_bbox_id_dict exists. "
                                    "The bbox will be read as category id: {}".format(shape['label'], cls))

            pts = shape['points']
            x0, y0 = pts[0]
            x1, y1 = pts[1]
            x0 = float(x0);y0 = float(y0);x1 = float(x1);y1 = float(y1)
            x0, x1 = min(x0, x1),max(x0, x1); y0, y1 = min(y0, y1),max(y0, y1)
            x0 = max(0,x0); y0 = max(0,y0); x1 = min(jsonfileobj["imageWidth"]-1,x1); y1 = min(jsonfileobj["imageHeight"]-1,y1)
            width = x1-x0; height = y1-y0
            dataset['annotations'].append({'area':width*height,
                                           'bbox':[x0, y0, width, height],
                                           'category_id': cls,
                                           'id': id_annotation,  # suppose to be
                                           'image_id': id_img,
                                           'iscrowd' : 0,
                                           'segmentation':[[x0,y0,x1,y0,x1,y1,x0,y1]]
                                           })
            id_annotation+=1
            #end for shape in shapes_list


        id_img += 1
        #end for jsonfilename in imglist

    # Add entry/entries to "categories" field & Check inclusiveness between the provided categories and categories of existing bboxes.
    normal_bbox_id_list = list(set(normal_bbox_id_list))
    if len(args.categories)>0:
        for normal_id in normal_bbox_id_list:
            if normal_id not in args.categories.keys():
                dataset['categories'].append({'supercategory': str(normal_id), 'id': normal_id, 'name': str(normal_id)})
                logging.warning("Category id {} which appears in label file(s) doesn't appear in args.categories. "
                                "Still will be added to dataset's 'categories' field.".format(normal_id))

        for error_name,error_id in error_bbox_id_dict.items():
            if error_id not in args.categories.keys():
                dataset['categories'].append({'supercategory': error_name, 'id': error_id, 'name': error_name})
                logging.warning("Category id {} newly generated while reading label file(s) isn't in args.categories. "
                                "Still will be added to dataset's 'categories' field.".format(error_id))
        for id, name in args.categories.items():
            dataset['categories'].append({'supercategory': name, 'id': id, 'name': name})
    else:
        for normal_id in normal_bbox_id_list:
            dataset['categories'].append({'supercategory': str(normal_id), 'id': normal_id, 'name': str(normal_id)})

        for error_name,error_id in error_bbox_id_dict.items():
            dataset['categories'].append({'supercategory': error_name, 'id': error_id, 'name': error_name})


    # Write dict into dataset json file
    dataset_name = args.root_path.split("/")[-1]+'.json'
    dataset_json_file_path = os.path.join(args.output_path,dataset_name)
    with open(dataset_json_file_path,'w') as f:
        json.dump(dataset,f)


    logging.info("read_json.py: Processing finished. Images are written into:{}. "
                 "COCO-API Dataset Json file is written into:{}".format(args.output_path,args.output_path))
    return


# \brief  parse args
# \usage  used in __main__
def parse_args():
    parser = argparse.ArgumentParser(description="Read & Convert Json Format Files Produced by LabelImg")

    # Parameters to set
    parser.add_argument("--root-path",default = "/home/user/pictures/jsonfilespath", help = "root path of files")
    parser.add_argument("--output-path", default="/home/user/pictures/output", help="output path of config file & images")

    parser.add_argument("--recursive-depth",type = int, default = 0, help = "depth of root path to files, if files are on the root path, the value should be 0")
    parser.add_argument("--label-type", default = "points", help = "[points],...")

    parser.add_argument("--categories", default={1111:'meter'}, type=dict, help="it should be as {0:'dog', 1:'cat'} the key should be category id(int), and the value should be category name(str)")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # Read file(s) and write into coco-like dataset
    write_as_cocolike(args)
