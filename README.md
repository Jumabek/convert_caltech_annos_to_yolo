# convert_caltech_annos_to_yolo
This script will convert caltech annotations into yolo format

You first need to extract .seq files into images and annotations into text files using either 
- evaluation tool from http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/  and extract seq into images directory using dbExtract.m
- with python https://github.com/mitmul/caltech-pedestrian-dataset-converter

next use convert_caltech_annos_to_yolo script to convert annotations into yolo format by providing

```python
parser.add_argument('-images_root', default = 'F:\dataset\CaltechPedestrians\\train_images', # This dir name I cannot change cuz it will cause confusion 
                        help='path to root of the images dir\n', )
    parser.add_argument('-annos_root', default = 'F:\dataset\CaltechPedestrians\\txt_annotations',  
                        help='path to root of the annotations dir ')
    parser.add_argument('-yolo_annos_root', default = 'F:\\dataset\\CaltechPedestrians\\fullBB-sample\\train_labels', 
                        help='path to root of simple annotations dir')
    parser.add_argument('-filelist', default = 'F:\\dataset\\CaltechPedestrians\\fullBB-sample\\filelist_train.txt', 
                        help='name of the file that contain paths to images.')
```
