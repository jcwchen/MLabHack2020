import mlab_model
import argparse
#import onnx


def validate_onnx_file():
    pass
def validate_onnx_graph():
    pass
def main():
    parser = argparse.ArgumentParser(description='load onnx from github')
    parser.add_argument('--url', type=str,
                        help='github url which user provides')
    args = parser.parse_args()                   
    print(args.url)

if __name__ == '__main__':
    main()
