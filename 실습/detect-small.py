# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os

from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import collections
from PIL import Image

from utils4smalls import tiles_location_gen, non_max_suppression, draw_object, reposition_bounding_box, set_resized_input

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='./out.mp4')
    parser.add_argument('--length', type=int, default=7)

    parser.add_argument('--tile_sizes', required=True)
    parser.add_argument('--tile_overlap', type=int, default=15)
    parser.add_argument('--iou_threshold', type=float, default=0.1)
    parser.add_argument('--score_threshold', type=float, default=0.5)

    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)

    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
    if cap.isOpened():
        frame_width = int(cap.get(3))   
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) #frame rate
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #output video format arguments
    out = cv2.VideoWriter(args.output,fourcc, fps, (frame_width, frame_height))
    frames = fps * args.length
    ##### add this block #####
    img_size = (frame_width, frame_height)
    tile_sizes = [
        list(map(int, tile_size.split('x'))) for tile_size in args.tile_sizes.split(',')
    ]
    ##########################


    while cap.isOpened() and frames>0:
        ret, frame = cap.read()
        if not ret:
            break
        
        im = Image.fromarray(frame).convert('RGB')  #Image.open('/home/mendel/h-new-land.jpg').convert('RGB')
        
        objects_by_label = dict()

        for tile_size in tile_sizes:
            for tile_location in tiles_location_gen(img_size, list(tile_size), args.tile_overlap):
                ##### fill this block #####
                #crop tile
                tile = im.crop(tile_location)
                _, scale = set_resized_input(interpreter, tile.size,
                      lambda size, img=tile: img.resize(size, Image.NEAREST))
                interpreter.invoke()
                #####################
                
                objs = get_objects(interpreter, args.score_threshold, scale)
                
                for obj in objs:
                    bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
                    bbox = reposition_bounding_box(bbox, tile_location)
                    label = labels.get(obj.id,'') 
                    objects_by_label.setdefault(label, []).append(Object(label, obj.score, bbox))
        
        ##### fill this block #####
        for label, objects in objects_by_label.items():
            idxs = non_max_suppression(objects, args.iou_threshold)
            for idx in idxs:
                draw_object(frame, objects[idx])
        ###########################
       
        ##### add this block #####
        out.write(frame) #change
        frames -=1
        print(f'{int(frames)} frames left!')
        ##########################

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
