import json
import os

input_file = 'base_de_dados/_annotations.coco_train.json'
out_file = 'simple_coco_ann_train.json'

class JsonManager:
    def __init__(self, in_file, out_file):
        self.path_in_file = in_file
        self.path_out_file = out_file
        self.data_in = None
        self.simplified_data = {
            'images': [],
            'annotations': []
        }

    def read_json(self):
        with open(self.path_in_file, 'r') as f:
            self.data_in = json.load(f)

    def processes_data(self):
        for img in self.data_in['images']:
            simplified_image = {
                'img_id': img['id'],
                'file_name': img['file_name'],
                'height': img['height'],
                'width': img['width']
            }
            self.simplified_data['images'].append(simplified_image)

        image_to_best_annotation = {}
        for ann in self.data_in['annotations']:
            image_id = ann['image_id']
            area = ann.get('area', 0)

            if image_id not in image_to_best_annotation or area > image_to_best_annotation[image_id].get('area', 0):
                image_to_best_annotation[image_id] = ann

        for image_id, ann in image_to_best_annotation.items():
            simplified_ann = {
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area']
            }
            self.simplified_data['annotations'].append(simplified_ann)

    def write_json(self):
        with open(self.path_out_file, 'w') as f:
            json.dump(self.simplified_data, f, indent=2)

jsonmanager = JsonManager(input_file, out_file)
jsonmanager.read_json()
jsonmanager.processes_data()
jsonmanager.write_json()
