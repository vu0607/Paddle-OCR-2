from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import imgaug
import imgaug.augmenters as iaa
import imageio
import os
import argparse


class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
            else:
                return getattr(iaa, args[0])(
                    *[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{
                k: self.to_tuple_if_list(v)
                for k, v in args['args'].items()
            })
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment():
    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            augmenter_args = [{
                'type': 'Fliplr',
                'args': {
                    'p': 0.5
                }
            }, {
                'type': 'Affine',
                'args': {
                    'rotate': [-10, 10]
                }
            }, {
                'type': 'Resize',
                'args': {
                    'size': [0.5, 3]
                }
            }]
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        image = data['image']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['image'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(
                keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly


if __name__ == '__main__':
    img_path = './input_image/CMND.jpg'
    input_image = imageio.imread(img_path)
    data = {'image': input_image, 'polys': ''}
    arg = [{
                'type': 'Fliplr',
                'args': {
                    'p': 0.25
                }
            }, {
                'type': 'Affine',
                'args': {
                    'rotate': [-10, 10]
                }
            }, {
                'type': 'MotionBlur',
                'args': {
                    'k': [5, 11], 'angle' : [0,360]
                }
            }, {
                'type': 'GaussianBlur',
                'args': {
                    'sigma': [0.5, 2.5]
                }
            }, {
                'type': 'Spatter',   #Muc sigma max = 2.0
                'args': {
                    'severity': [1, 1]
                }
            }, {
                'type': 'GammaContrast',
                'args': {
                    'gamma': [0.5, 2.5], 'per_channel': True
                        }
            }]
    class_aug = IaaAugment(arg)
    img = class_aug(data)
    imgaug.imshow(img['image'])