import albumentations as albu


def get_transforms(size: tuple[int,int]):
    # augs = {'weak': albu.Compose([albu.HorizontalFlip(),
    #                               ]),
    #         'geometric': albu.OneOf([albu.HorizontalFlip(p=1),
    #                                  albu.ShiftScaleRotate(p=1),
    #                                  albu.Transpose(p=1),
    #                                  albu.OpticalDistortion(p=1),
    #                                  albu.ElasticTransform(p=1),
    #                                  ])
    #         }
    #
    # aug_fn = augs['geometric']
    # crop_fn = {'random': albu.RandomCrop(size, size, p=1),
    #            'center': albu.CenterCrop(size, size, p=1)}['random']
    
    effect = albu.OneOf([albu.MotionBlur(blur_limit=21, p=1),
                         albu.RandomRain(p=1),
                         albu.RandomFog(p=1),
                         albu.RandomSnow(p=1)])
    motion_blur = albu.MotionBlur(blur_limit=55, p=1)

    resize = albu.Resize(height=size[0], width=size[1])

    pipeline = albu.Compose([resize, motion_blur], additional_targets={'target': 'image'})

    pipforblur = albu.Compose([effect])

    def process(a, b):
        f = pipforblur(image=a)
        r = pipeline(image=f['image'], target=b)
        return r['image'], r['target']

    return process


def get_transforms_fortest(size):
    resize = albu.Resize(height=size[0], width=size[1])

    effect = albu.OneOf([albu.MotionBlur(p=1),
                         albu.RandomRain(p=1),
                         albu.RandomFog(p=1),
                         albu.RandomSnow(p=1)])
    motion_blur = albu.MotionBlur(blur_limit=51, p=1)

    pipeline = albu.Compose([resize, effect, motion_blur], additional_targets={'target': 'image'})

    def process(a, b):
        r = pipeline(image=a, target=b)
        return r['image'], r['target']

    return process


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process
