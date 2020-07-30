import onnxruntime
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def super_image(input_filename, output_filename):
    image_shape = (224, 224) # fixed for this model
    batch_size = 1 # test single image

    # inference graph
    ort_session = onnxruntime.InferenceSession("SuperResolution/model/super-resolution-10.onnx")
    x = torch.randn(batch_size, 1, image_shape[0], image_shape[1], requires_grad=True)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    img = Image.open(input_filename)

    # resize image into 224 * 224
    img = Image.open(input_filename)
    resize = transforms.Resize([image_shape[0], image_shape[1]])
    img = resize(img)
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)
    
    # inference with input image
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    
    # process output image
    img_out_y = Image.fromarray(numpy.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    final_img.save(output_filename)


def main():
    parser = argparse.ArgumentParser(description='demo Super Resolution')
    # for nn.Module validation
    parser.add_argument('--input', action='store', default='cat_224x224.jpg', type=str)
    parser.add_argument('--output', action='store', default='super_cat.jpg', type=str)

    args = parser.parse_args()
    super_image(args.input, args.output)


if __name__ == '__main__':
    main()
