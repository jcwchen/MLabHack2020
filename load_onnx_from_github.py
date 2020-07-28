import mlab_model
import argparse
import pickle
import json
#import onnx


def validate_onnx_file():
    pass
def validate_onnx_graph():
    with open("model.pkl", 'rb') as f:
        pkl = pickle.load(f)
    
def main():
    parser = argparse.ArgumentParser(description='load onnx from github')
    parser.add_argument('--url', type=str,
                        help='github url which user provides')
    args = parser.parse_args()                   
    output = {'url': args.url}
    print(output)

if __name__ == '__main__':
    main()
