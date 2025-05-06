from .aro_datasets import  Controlled_Images, COCO_QA, VG_QA, VSR

def get_dataset(dataset_name, image_preprocess=None, text_perturb_fn=None, image_perturb_fn=None, download=False, *args, **kwargs):
    """
    Helper function that returns a dataset object with an evaluation function. 
    dataset_name: Name of the dataset.
    image_preprocess: Preprocessing function for images.
    text_perturb_fn: A function that takes in a string and returns a string. This is for perturbation experiments.
    image_perturb_fn: A function that takes in a PIL image and returns a PIL image. This is for perturbation experiments.
    download: Whether to allow downloading images if they are not found.
    """
    
    if dataset_name== "VSR":
        from .datasets import get_vsr
        return get_vsr(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")