from os import listdir
from os.path import isfile, join
import argparse
import cv2
import numpy as np
import sys
import os
import shutil
import copy
     
'''
for every set%0.2d
    for every V%0.3d
        convert_video_images(params)
'''
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir,name))]

num_person = 0
num_people = 0
classes_indices = {"person":0,"dont-care":-1}

def is_reasonable_for_train(anno,v_anno):
    [c,x,y,w,h] = anno
    [cv,xv,yv,wv,hv] = v_anno

    #check if annotation is reasonable (pedestrians that are at least 50 pixels tall and at least 65% visiable) http://kaiminghe.com/publications/eccv16ped.pdf
    # http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/PAMI12pedestrians.pdf
    
    if int(hv)>30 and float(wv)*float(hv)/(float(w)*float(h))>=0.65:
        return True
    return False
 
def is_reasonable_for_test(anno,v_anno):
    [c,x,y,w,h] = anno
    [cv,xv,yv,wv,hv] = v_anno

    #check if annotation is reasonable (pedestrians that are at least 50 pixels tall and at least 65% visiable) http://kaiminghe.com/publications/eccv16ped.pdf
    # http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/files/PAMI12pedestrians.pdf

    
    if int(hv)>30 and  float(wv)*float(hv)/(float(w)*float(h)+0.00001)>=0.65:
        return True
    return False
 
def read_annotations(filename):
    f = open(filename,'r')
    annotations = []
    v_annotations = []
    dummy_line = f.readline()

    global num_people
    global num_person

    while True:
        line=f.readline()
        if line=="":
            #print "End of line reached"
            break
        args = line.split(" ")
        annotation = []
        v_annotation = []
        
        if args[0].startswith("person"):
            cls_id = classes_indices['person']
            num_person = num_person +1
        elif  args[0].startswith("people"):
            cls_id = classes_indices['person'] 
        
        annotation.append(cls_id)
        annotation.append(args[1])
        annotation.append(args[2])
        annotation.append(args[3])
        annotation.append(args[4])

        oc_index = 5
        if args[oc_index] == '1' :
            v_annotation.append(cls_id)
            v_annotation.append(args[oc_index + 1])
            v_annotation.append(args[oc_index + 2])
            v_annotation.append(args[oc_index + 3])
            v_annotation.append(args[oc_index + 4])
        else:     
            v_annotation = copy.copy(annotation)

        if len(annotation)>0 :
            print 'checking for reasonable {} with {}'.format(filename,annotation)
            if is_reasonable_for_test(annotation,v_annotation):
                annotations.append(annotation) 
        
    return annotations

def draw_annos(image,annos):
    for i in range(len(annos)):
        bbox_str = annos[i]
        bbox_int = [int(v) for v in bbox_str]
        [c,x,y,w,h] = bbox_int    

        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255))

def write_yolo_annos(filename, size, annos, filestream, image_file):
    
    if len(annos)==0:
        return

    f = open(filename,"w")
    filestream.write("%s \n"%image_file)        

    for i in range(len(annos)):
        [cls_id,x,y,w,h] = annos[i]
        if cls_id=='-1':
            continue
    
        xmin = float(x)
        xmax = xmin + float(w)
        ymin = float(y)
        ymax = ymin + float(h)

        b = (xmin, xmax, ymin, ymax)
        bb = convert(size, b)
        f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
    
def convert_video_images(images_dir,annos_dir,yolo_annos_dir, fileliststream):
    
    global num_people
    global num_person

    print "images dir you provided {}".format(images_dir)
    print "annos dir you provided is {}".format(annos_dir)
    print "yolo annos dir you provided is {}".format(yolo_annos_dir)
    
    imagefiles = [f for f in listdir(images_dir) if isfile(join(images_dir, f))]
    
    annotations = [f for f in listdir(annos_dir) if isfile(join(annos_dir,f))]
    annotations = np.array(annotations)
    annotations = np.sort(annotations)
    
    #filtering '*.jpg' files
    imagefiles = [image_file for image_file in imagefiles if image_file[image_file.rfind('.')+1:]=='jpg']
    imagefiles = np.array(imagefiles)
    imagefiles = np.sort(imagefiles)
    
    #skip = 30 # for caltech 1x
    skip = 3 # for caltech 10x
    
    for i in range(skip-1,imagefiles.shape[0],skip):
        image_file = join(images_dir,imagefiles[i])
        im = cv2.imread(join(images_dir,imagefiles[i]))
        if im is None:
            continue
        annos = read_annotations(join(annos_dir,annotations[i]))
    
        (h,w) = im.shape[:2]
        size = (w,h)
        write_yolo_annos(join(yolo_annos_dir, annotations[i]),size,annos,fileliststream,image_file)
        draw_annos(im,annos)
        cv2.imshow('image',im)
        cv2.waitKey(1000/120)

    print "num_people = %d\t num_person = %d\n"%(num_people,num_person)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_root', default = 'F:\dataset\CaltechPedestrians\\train_images', # This dir name I cannot change cuz it will cause confusion 
                        help='path to root of the images dir\n', )
    parser.add_argument('-annos_root', default = 'F:\dataset\CaltechPedestrians\\txt_annotations',  
                        help='path to root of the annotations dir ')
    parser.add_argument('-yolo_annos_root', default = 'F:\\dataset\\CaltechPedestrians\\fullBB-sample\\train_labels', 
                        help='path to root of simple annotations dir')
    parser.add_argument('-filelist', default = 'F:\\dataset\\CaltechPedestrians\\fullBB-sample\\filelist_train.txt', 
                        help='name of the file that contain paths to images.')
    
    args = parser.parse_args()
    
    sets = get_immediate_subdirectories(args.images_root)
    print sets

    #clean the directory that we want to generete annotations to
    if os.path.exists(args.yolo_annos_root):
        shutil.rmtree(args.yolo_annos_root)

    #create empty dir
    os.makedirs(args.yolo_annos_root)   


    fileliststream = open(args.filelist,'w')

    for set in sets:
        print "Getting immideate subdirectories from %s ", join(args.images_root, set)
        vbbs = get_immediate_subdirectories(join(args.images_root, set))
        for vbb in vbbs:
            os.makedirs(join(args.yolo_annos_root,set,vbb))
            print "Passing %s"%('-images %s -annos %s -yolo_annos %s'%(join(args.images_root,set,vbb), join(args.annos_root,set,vbb), join(args.yolo_annos_root,set,vbb)))
            convert_video_images(join(args.images_root,set,vbb), join(args.annos_root,set,vbb), join(args.yolo_annos_root,set,vbb),fileliststream)

if __name__=="__main__":
    main(sys.argv)
