import numpy as np 
import torch
import torchvision
import random


def ids_by_label(ids, labels_list, target, labels_set, shuffle):
    '''
    Gets ids segregated according to target label.

    --Input:
        ids > list of IDS
        labels_list > list of labels correspondong to the ids. Each element ranges from 0-9
        label > integer (0-9) based on which labels are separated into two lists
        shuffle > shuffle ids and labels in the labels_list

    --Output:   
        tuple of two lists.
        first list contains ids corresponding to 'label', second list contains ids NOT corresponding to 'label'
    '''
    if not isinstance(shuffle,bool):
        raise TypeError("shuffle should be a bool")
    pairs = list(zip(ids,labels_list))
    if labels_set != 'all':
       pairs = [(_id, label) for _id,label in pairs if label.item() in labels_set]
    if shuffle:
        random.shuffle(pairs)
    return [_id for _id,_label in pairs if _label.item()==target], [_id for _id,_label in pairs if _label.item()!=target]


def check_if_target(indices, labels, target):
    '''
    Checks if the list of indices consists label equal to a defined target
    --Input:
        indices > list of indices
        labels > labels corresponding to the indices
        target > integer (0-9). target to check for

    --Output:
        bool > True of the target is present, false if it is not
    '''
    for idx in indices:
        if labels[idx].item() == target:
            return True 
    return False


def make_bags(target_ids, non_target_ids, labels, bag_size, num_bags, prob, target):
    '''
    Make a generator object for bag generation. A bag consists of N images. 
    The bag belongs to one of two classes. In the first class, the bag has ATLEAST ONE image labelled target
    In the second class, the bag has no image with target label

    --Inputs:
        target_ids > list of ids having label equal to target
        non_target_ids > list of ids having label not equal to target
        labels > list of labels according to ids
        bag_size > number of instances in a bag
        num_bags > number of bags to be generated
        prob > probability that the bag has the target class
        target > int (0-9). target label

    --Output:
        (list_one, list_two)
        list_one > list of size = bag_size. Each element is an id. 
        list_two > list of size one. [1] or [0]

    '''
    full_list = target_ids + non_target_ids 
    for _ in range(num_bags):
        random.shuffle(full_list)
        random.shuffle(non_target_ids)
        toss = np.random.rand()
        if toss <= prob:
            random.shuffle(target_ids)
            rest = full_list[:bag_size-1]
            rest.append(target_ids[0])
            assert check_if_target(rest,labels,target), "9 should be present " ### check presence of target
            yield rest, [1]
        else:
            rest = non_target_ids[:bag_size]
            assert not check_if_target(rest,labels,target), "9 should not be present" ### check absence of target
            yield rest, [0]



def get_patches(indices,images):
    '''
    Get tensor of images in a bag
    --Input:
        indices > list of index in a bag
        images > total set of images

    --Output:
        tensor > shape = (bag_size, *image_dim)
    '''
    patch_set = []
    for idx in indices:
        patch_set.append(images[idx])

    patch_set = torch.stack(patch_set,0)
    return patch_set

def num_labels_in_bag(indices,labels_list, label):
    '''
    Gets number of label instances in the list of indices
    --Input:
        indices > list of ids 
        labels_list > list of labels (0-9)
        label > int (0-9)

    --Output
        int > count of labels in the list
    '''
    return sum([1 for idx in indices if labels_list[idx] == label])

if __name__ == '__main__':
    mnist = torchvision.datasets.MNIST(root='',download=True)
    images = mnist.data #shape = (60000,28,28)
    labels = mnist.targets #shape = (60000)
    ids = np.arange(len(labels))

    BAG_SIZE = 300
    PROB = 0.4
    NUM_BAGS = 1000
    TARGET = 9

    target_ids, non_target_ids = ids_by_label(ids,labels,9,'all', True)
    

    ############################# TESTING ##################################
    for _id in target_ids:
        if labels[_id].item() != TARGET:
            raise ValueError("id: {} is not target".format(labels[_id]))

    for _id in non_target_ids:
        if labels[_id].item() == TARGET:
            raise ValueError("id: {} is equal to target".format(labels[_id]))
    #########################################################################

    bag_gen = make_bags(target_ids, non_target_ids, labels, BAG_SIZE, NUM_BAGS, PROB, TARGET)
    sample_bag, sample_label = next(bag_gen)
    print(sample_bag.__len__())
    patch_set = get_patches(sample_bag,images)
    print(patch_set.shape)

