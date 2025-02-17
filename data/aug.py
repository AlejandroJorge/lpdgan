import albumentations as albu


def get_transforms(height, width):
    augs = {'weak': albu.Compose([albu.HorizontalFlip(),
                                  ]),
            'geometric': albu.OneOf([albu.HorizontalFlip(p=1.0),
                                     albu.ShiftScaleRotate(p=1.0),
                                     albu.Transpose(p=1.0),
                                     albu.OpticalDistortion(p=1.0),
                                     albu.ElasticTransform(p=1.0),
                                     ])
            }

    aug_fn = augs['geometric']
    crop_fn = {'random': albu.RandomCrop(height=height, width=width, p=1.0),
               'center': albu.CenterCrop(height=height, width=width, p=1.0)}['random']

    effect = albu.OneOf([albu.MotionBlur(blur_limit=21, p=1.0),
                         albu.RandomRain(p=1.0),
                         albu.RandomFog(p=1.0),
                         albu.RandomSnow(p=1.0)])
    motion_blur = albu.MotionBlur(blur_limit=55, p=1.0)

    resize = albu.Resize(height=height, width=width)

    pipeline = albu.Compose([resize], additional_targets={'target': 'image'})

    pipforblur = albu.Compose([effect])

    def process(a, b):
        f = pipforblur(image=a)
        r = pipeline(image=f['image'], target=b)
        return r['image'], r['target']

    return process


def get_transforms_fortest(size):
    resize = albu.Resize(height=size[0], width=size[1])

    effect = albu.OneOf([albu.MotionBlur(p=1.0),
                         albu.RandomRain(p=1.0),
                         albu.RandomFog(p=1.0),
                         albu.RandomSnow(p=1.0)])
    motion_blur = albu.MotionBlur(blur_limit=51, p=1.0)

    pipeline = albu.Compose([resize], additional_targets={'target': 'image'})

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
