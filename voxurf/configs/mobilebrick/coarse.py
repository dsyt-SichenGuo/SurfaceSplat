_base_ = '../coarse.py'

data = dict(
    dataset_type='mobilebrick',
    cameras_name='cameras.npz',
    num_img=10,                   # 10 evenly spaced input views
)

surf_train = dict(
    normal_mask=False,            # normal loss on the whole image (not restricted to the object mask)
    normal_cos_reduce='all',
)
