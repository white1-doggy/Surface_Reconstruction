import os
import ants
import scipy
import numpy as np

from tqdm import tqdm
from IPython import embed
from skimage import measure
from collections import Counter


def _find_majority_element(arr, idx):
    # 移除0并统计出现次数
    filtered_arr = [num for num in arr if num != 0 and num != 100 and num!=idx]
    count = Counter(filtered_arr)

    # 找到出现最多的数
    if count:
        majority_element = max(count, key=count.get)  # 获取出现次数最多的元素
        return majority_element, count[majority_element]
    else:
        return None, 0  # 如果全是0，返回None或其他适当值


label_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 59, 60, 61, 62, 75, 76, 89, 90,
               93, 94]


def _seg_to_label(seg):
    '''
    TODO: Labeling each single annotation in one image (single label to multiple label)
    :param seg: single label annotation (numpy data)
    :return: multiple label annotation
    '''
    labels, num = measure.label(seg, return_num=True)
    return labels, num


def _select_top_k_region(img, k=2):
    '''
    TODO: Functions to select top k connection regions
    :param img: numpy array with multiple regions
    :param k: number of selected regions
    :return: selected top k region data
    '''
    # seg to labels
    labels, nums = _seg_to_label(img)
    rec = list()

    for idx in range(1, nums + 1):
        subIdx = np.where(labels == idx)
        rec.append(len(subIdx[0]))
    rec_sort = rec.copy()
    rec_sort.sort()

    rec = np.array(rec)
    index = np.where(rec >= rec_sort[-k])[0]
    index = list(index)

    for idx in index:
        labels[labels == idx + 1] = 1000000

    labels[labels != 1000000] = 0
    labels[labels == 1000000] = 1

    return labels


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _dk_refine(source, target, item):
    dk_path = os.path.join(source, item, 'dk-struct.nii.gz')
    dk_refined_path = os.path.join(target, item, 'dk-struct.nii.gz')

    origin, spacing, direction, dk = _ants_img_info(dk_path)
    for idx in label_index:
        tmp = dk.copy()
        tmp[tmp != idx] = 0
        tmp[tmp == idx] = 1
        labels, labels_number = _seg_to_label(tmp)

        if labels_number == 1:
            continue
        else:
            tmp_mask = _select_top_k_region(tmp, k=1)
            tmp[tmp_mask != 0] = 0
            tmp_label, tmp_num = _seg_to_label(tmp)
            for tmp_idx in range(1, tmp_num + 1):
                tmp_mask = tmp_label.copy()
                tmp_mask[tmp_mask != tmp_idx] = 0
                tmp_mask[tmp_mask == tmp_idx] = 1
                tmp_mask_enlarged = scipy.ndimage.morphology.binary_dilation(tmp_mask)
                tmp_mask_enlarged[tmp_mask != 0] = 0
                neighborhood_index = np.where(tmp_mask_enlarged != 0)
                neighborhood = dk[neighborhood_index]
                majority_element, count_majority_element = _find_majority_element(neighborhood, idx)

                if majority_element == None:
                    continue
                else:
                    dk[tmp_mask != 0] = majority_element
    dk = ants.from_numpy(dk, origin, spacing, direction)
    ants.image_write(dk, dk_refined_path)


def update(pbar, result):
    pbar.update()


def error_back(err):
    print(err)


if __name__ == '__main__':
    from multiprocessing import Pool
    from tqdm import tqdm
    from IPython import embed

    source = '/home_data/home/lianzf2024/test'
    target = '/home_data/home/lianzf2024/test'

    file_list = os.listdir(source)
    file_list.sort()

    pool_num = 8
    pool = Pool(pool_num)
    pbar = tqdm(total=len(file_list))
    pbar.set_description('Data Reorient')
    call_fun = lambda *args: update(pbar, *args)

    for item in file_list:
        kwargs = {
            'source':source,
            'target':target,
            'item':item
            }
        pool.apply_async(_dk_refine, args=(), kwds=kwargs, callback=call_fun, error_callback=error_back)

    pool.close()
    pool.join()

