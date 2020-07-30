# SuperResolutionNet

### Github Link
```https://github.com/jcwchen/MLabHack2020/tree/master/SuperResolution```

### Onnx File Name
```super-resolution-10```

### Include Training and Inference
```True```

### Python/Module File name
```SRNet```

### Module Class Name
```SuperResolutionNet```

### Model Name
```Super Resolution Net```

### Category Name
```Generative Model```

### Tasks
```Image Quality Refinement```

### Cover Image
```https://upload.wikimedia.org/wikipedia/commons/3/3a/FBALM_DNA_superresolution_HeLa_cell_nucleus.png```

### Input Description
Image input sizes are dynamic. The inference was done using jpg image.

### Preprocessing Description
Images are resized into (224x224). The image format is changed into YCbCr with color components: greyscale ‘Y’, blue-difference ‘Cb’, and red-difference ‘Cr’. 
Once the greyscale Y component is extracted, it is then passed through the super resolution model and upscaled. Sample code:
```python
from PIL import Image
from resizeimage import resizeimage
import numpy as np

orig_img = Image.open('IMAGE_FILE_PATH')
img = resizeimage.resize_cover(orig_img, [224,224], validate=False)
img_ycbcr = img.convert('YCbCr')
img_y_0, img_cb, img_cr = img_ycbcr.split()
img_ndarray = np.asarray(img_y_0)
img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
img_5 = img_4.astype(np.float32) / 255.0
img_5
```

### Output Description
The model outputs a multidimensional array of pixels that are upscaled. Output shape is [batch_size,1,672,672]. The second dimension is one because only the (Y) intensity channel was passed into the super resolution model and upscaled.

### Postprocessing Description
Postprocessing involves converting the array of pixels into an image that is scaled to a higher resolution. The color channels (Cb, Cr) are also scaled to a higher resolution using bicubic interpolation. Then the color channels are combined and converted back to RGB format, producing the final output image.
Sample code:
```python
from PIL import Image
import matplotlib.pyplot as plt

final_img = Image.merge(
"YCbCr", [
    img_out_y,
    img_cb.resize(img_out_y.size, Image.BICUBIC),
    img_cr.resize(img_out_y.size, Image.BICUBIC),
]).convert("RGB")
plt.imshow(final_img)
```

### Hyperparameter Description
```
N/A
```

### Dataset Name
```BSD300 Dataset```

### Dataset URL
```http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz```

### Paper Authors
```N/A```

### Paper Link
```N/A```

### Evaluation Metrics
```N/A```

### Evaluation Results
```N/A```

### Training/Validation Loss Graph
```N/A```
