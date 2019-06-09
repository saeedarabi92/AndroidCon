from PIL import Image
import os

def main(folder_1):
    for dirpath_1,_,filenames_1 in os.walk(folder_1):
        for f_1 in filenames_1:
            if f_1 != ".DS_Store":
                abspath_1 = os.path.abspath(os.path.join(dirpath_1, f_1))
                img = Image.open(abspath_1)
                # resize of the image
                img = img.resize((640,480))
                # rotation of the image
                #img = img.rotate(90)
                img.save("processed_image/"+f_1.split('.')[0]+'.jpg')
                print('Image',f_1, 'preprocessing is done!' )

if __name__ == '__main__':

    Package_dir = os.path.dirname(os.path.realpath(__file__))
    folder_1 = Package_dir + '/test_image'
    main(folder_1)
