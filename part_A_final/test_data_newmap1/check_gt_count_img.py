from PIL import Image
import os
import glob
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import scipy.io as io


src_folder = "./images_p"
tar_folder = "./ground_truth"
# tar_folder = "./ground_truth_new"
# tar_folder = "./ground_truth"
# tar_folder = "./ground_truth_old"
tar_folder_gt = "./ground_truth_p_cut"
backup_folder = "backup"


sum1=[]
sum2=[]
def handleImage(filename, tar):
    # # if os.path.exists(os.path.join(tar, filename)):
    # #     return
    # img = Image.open(os.path.join(src_folder, filename))
    # if img.mode != "RGB":
    #     img = img.convert("RGB")
    # # img = Image.open(os.path.join(filename))
    # # print(img.mode )
    # width, height = img.size
    # print('height%d,width%d'%(height,width))
    #
    # left = boundaryFinder(img, 0, width / 2, vCheck)
    # right = boundaryFinder(img, width - 1, width / 2, vCheck)
    # top = boundaryFinder(img, 0, height*2 / 3, hCheck)
    # bottom = boundaryFinder(img, height - 1,height* 2/ 3, hCheck)
    #
    # rect = (left, top, right, bottom)
    # print(rect)
    # region = img.crop(rect)
    # region.save(os.path.join(tar, filename))

    # gt_path = os.path.join(src_folder, filename).replace('.jpg', '.h5').replace('images_p', 'ground_truth_p')
    # gt_path = os.path.join(src_folder, filename).replace('.jpg', '.h5').replace('images_p', 'ground_truth_p_cut')
    # gt_path = os.path.join(tar_folder, filename)
    gt_path=filename
    print(gt_path)
    gt_file = h5py.File(gt_path)
    image_gt = np.asarray(gt_file['density'])
    # print(filename)
    # print('image_gt:',np.sum(image_gt))

    # gt_path1 = gt_path.replace('ground_truth_new','../test_data/ground_truth')
    # gt_path1 = gt_path.replace('ground_truth', '../test_data_15/ground_truth')
    gt_path1 = gt_path.replace('ground_truth', '../test_data/ground_truth')
    # print(gt_path)
    gt_file1 = h5py.File(gt_path1)
    image_gt1 = np.asarray(gt_file1['density'])

    # gt_mat_path=gt_path.replace('ground_truth','../test_data/ground_truth').replace('IMG_','GT_IMG_').replace('.h5', '.mat')
    # mat=io.loadmat(gt_mat_path)
    # gt = mat["image_info"][0, 0][0, 0][0]

    img_path=gt_path.replace('ground_truth','images').replace('h5','jpg')
    # img_path = gt_path.replace('ground_truth_new', 'images').replace('h5', 'jpg')
    img_1=np.array(Image.open(img_path)) #.convert('L'))
    img_2=img_1.copy()
    # img_2=img_1*image_gt
    # img_3 = img_1 * image_gt1
    # img_2 = img_1 * image_gt1
    yuzhi=1e-5
    mask_1 = (image_gt < yuzhi)
    mask_2= (image_gt1 < yuzhi)
    img_1[mask_1,:]=0
    img_2[mask_2, :] = 0

    plt.subplot(2, 2, 1)
    plt.imshow(image_gt, cmap=CM.jet)
    plt.subplot(2, 2, 2)
    plt.imshow(image_gt1, cmap=CM.jet)


    plt.subplot(2, 2, 3)
    plt.imshow(img_1)
    plt.subplot(2, 2, 4)
    plt.imshow(img_2)
    # plt.subplot(2, 2,3)
    # plt.imshow(img_2)
    # plt.subplot(2, 2, 4)
    # plt.imshow(img_3)
    plt.show()

    # cha=np.sum(image_gt)-np.sum(image_gt1)
    # sum.append(cha)
    # print(cha)
    # print(np.mean(sum))
    # print(gt_path)
    # print('gt   map1    oldmap')
    # print(gt.shape[0],np.sum(image_gt),np.sum(image_gt1))
    # print(gt.shape[0]-np.sum(image_gt),'               ', gt.shape[0]-np.sum(image_gt1))
    # sum1.append(gt.shape[0]-np.sum(image_gt))
    # sum2.append(gt.shape[0]-np.sum(image_gt1))

    # image_gt = Image.fromarray(image_gt)
    # # image_gt=image_gt.convert("CMYK")
    # # image_gt.save(os.path.join(tar, filename))
    # region_gt=image_gt.crop(rect)
    # region_gt=np.array(region_gt)
    # #
    # # fig=plt.figure()
    # # ax=fig.add_subplot(121)
    # # ax.imshow(region)
    # # ax=fig.add_subplot(122)
    # # ax.imshow(np.array(region_gt), cmap=CM.jet)
    #
    #
    # with h5py.File(gt_path.replace('ground_truth_p', 'ground_truth_p_cut'),'w') as hf:
    #     hf['density'] = region_gt
    pass
def folderCheck(foldername):
    if foldername:
        if not os.path.exists(foldername):
            os.mkdir(foldername)
            print
            "Info: Folder \"%s\" created" % foldername
        elif not os.path.isdir(foldername):
            print
            "Error: Folder \"%s\" conflict" % foldername
            return False
    return True
    pass

def main():
    if folderCheck(tar_folder): #and folderCheck(src_folder) and folderCheck(tar_folder_gt):
        # for filename in os.listdir(tar_folder):
        # gt_paths=[
        #     '/media/adminn/文档/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data_newmap1/ground_truth/IMG_115.h5',
        #     '/media/adminn/文档/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data_newmap1/ground_truth/IMG_17.h5',
        #     '/media/adminn/文档/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data_newmap1/ground_truth/IMG_110.h5'
        # ]
        # for filename in gt_paths:
        for filename in glob.glob(os.path.join(tar_folder, '*.h5')):
            # if filename.split('.')[-1].upper() in ("JPG", "JPEG", "PNG", "BMP", "GIF"):
            # print(filename)
            # filename='IMG_0_-1_112.jpg'
            handleImage(filename,tar_folder)

            # os.rename(os.path.join(src_folder, filename), os.path.join(backup_folder, filename))
        pass


if __name__ == '__main__':
    main()
    print(np.mean(sum1),np.mean(sum2))
    print(np.sum(sum1), np.sum(sum2))