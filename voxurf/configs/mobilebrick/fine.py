_base_ = '../fine.py'

data = dict(
    dataset_type='mobilebrick',
    cameras_name='cameras.npz',
    num_img=10,
    load2gpu_on_the_fly=True,
)
