# Single Stage Detector

### Github Link
```https://github.com/jcwchen/MLabHack2020/edit/master/SSD/```

### Onnx File Name
```ssd-10```

### Include Training and Inference
```False```

### Model Name
```ssd```

### Category Name
```Computer Vision```

### Tasks
```Object Detection, Segmentation```

### Cover Image
```https://d3i71xaburhd42.cloudfront.net/8a471938c7841eba1ce30bbb876b736c5869fa7f/3-Figure1-1.png```

### Input Description
Image shape (1x3x1200x1200)

### Preprocessing Description
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
The transformation should preferrably happen at preprocessing. Sample code:
```python
from torchvision import transforms

def preprocess(img):   
    '''
    The function takes path to an image and returns processed tensor
    '''
    transform_fn = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0) # batchify
    
    return img
```

### Output Description
The model has 3 outputs. boxes: (1x'nbox'x4) labels: (1x'nbox') scores: (1x'nbox').

### Postprocessing Description
N/A

### Hyperparameter Description
```
N/A
```

### Dataset Name
```2017 COCO dataset```

### Dataset URL
```https://cocodataset.org/#home```

### Paper Authors
```Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.```

### Paper Link
```https://arxiv.org/pdf/1512.02325.pdf```

### Evaluation Metrics
```mAP (averaged over IoU) on 2017 COCO validation dataset```

### Evaluation Results
```0.195```

### Training/Validation Loss Graph
```N/A```
