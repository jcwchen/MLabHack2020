import onnxruntime
import torch
import torch.nn.functional as F
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

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
    img = img.view(1,3,224,224) # batchify
    
    return img

def mobilenet_image(input_filename):
    batch_size = 1 # test single image
    ort_session = onnxruntime.InferenceSession("MobileNet/model/mobilenetv2-7.onnx")
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    img = Image.open(input_filename)

    img_y = preprocess(img)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    img_out_y = F.softmax(torch.Tensor(img_out_y),dim=1)
    return torch.argmax(img_out_y)

def main():
    parser = argparse.ArgumentParser(description='demo MobileNet')
    parser.add_argument('--input', action='store', default='plane.jpg', type=str)

    args = parser.parse_args()
    pred = mobilenet_image(args.input)
    # check correctness: (class 896) warplane, military plane
    # reference: https://github.com/onnx/models/blob/2c4732abf3bb4890faed986b21853f7034f9979d/vision/classification/synset.txt#L896
    print("Predicted class:", pred.numpy()+1)

if __name__ == '__main__':
    main()
