_base_ = '../coarse.py'

data = dict(
    dataset_type='dtu',
    cameras_name='cameras_sphere.npz',
    num_img=5,                    # 5 evenly spaced input views
    mask_thres=0.8,
)

surf_train = dict(
    normal_mask=True,             # normal loss restricted to the object mask
    normal_cos_reduce='all',      # cosine term averaged over all rays
)
