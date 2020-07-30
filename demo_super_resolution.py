import onnxruntime
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def super_image(input_filename, output_filename):
    batch_size = 1 # test single image
    ort_session = onnxruntime.InferenceSession("SuperResolution/model/super-resolution-10.onnx")
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    img = Image.open(input_filename)

    resize = transforms.Resize([224, 224])
    img = resize(img)

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    img_out_y = Image.fromarray(numpy.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    final_img.save(output_filename)

def main(args):
    super_image("cat_224x224.jpg", "super_cat.jpg")

def main():
    parser = argparse.ArgumentParser(description='demo Super Resolution')
    # for nn.Module validation
    parser.add_argument('--input', action='store', default='cat_224x224.jpg', type=str)
    parser.add_argument('--output', action='store', default='super_cat.jpg', type=str)

    args = parser.parse_args()
    super_image(args.input, args.output)

if __name__ == '__main__':
    main()
