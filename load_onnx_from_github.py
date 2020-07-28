import mlab_model
import argparse
import pickle
import json
import requests
import json
import tempfile
import os.path as osp
import os
import onnx

github_username = ""
github_token = ""

def set_github_auth():
    try:
        import secret_config
        github_username = secret_config.GITHUB_USERNAME
        github_token = secret_config.GITHUB_TOKEN        
    except Exception as e:
        print(e)
        
def download_from_directory(url, save_directory, exclude_name={}):
    response = requests.get(url, auth=(github_username, github_token)).text
    result = json.loads(response)
    for files in result:
        if files['download_url']:
            file_path = osp.join(save_directory, files['name'])
            file_url = requests.get(files['download_url'])
            with open(file_path, 'wb') as f:
                f.write(file_url.content)
        elif files['type'] == 'dir' and files['name'] not in exclude_name:
            directory_path = osp.join(save_directory, files['name'])
            os.mkdir(directory_path)
            download_from_directory(files['url'].replace('?ref=master', ''), directory_path, exclude_name)

def find_onnx_file_in_directoy(path):
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.onnx'):
                return osp.join(path, filename)
    return None

def parse_github_url(url):
    api_url = convert2github_api(url) 
    print(api_url)
    with tempfile.TemporaryDirectory() as tmp:
        # download from model/
        model_directory_path = osp.join(tmp, 'model')
        os.mkdir(model_directory_path)
        download_from_directory(osp.join(api_url, 'model'), model_directory_path)
        onnx_path = find_onnx_file_in_directoy(model_directory_path).replace('\\', '/')
        print(onnx_path)
        try:
            #onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_path)
        except:
            # invalid onnx file
            return -1
        
    # valid GitHub url with valid file 
    return 1


def convert2github_api(url):
    """Add mounted path into local path
        Args:
            url (str): GitHub url. 
            e.g., https://github.com/jcwchen/MLabHack2020/tree/master/data/onnx/mnist
        Returns:
            str: GitHub API url e.g., 
            https://api.github.com/repos/jcwchen/MLabHack2020/contents
    """
    api_url = 'https://api.github.com/repos/'
    directory_path = ''
    i = -1
    for token in url.split('/'):
        print(token)
        i += 1
        # 0-2: https:// (useless)
        # 3-4: user name + repo name
        if 3 <= i <= 4:
            api_url += token + '/'
        # 4-5: branch tree (useless) 
        # >=7: directory path
        elif i >= 7:
            directory_path += token + '/'
    return api_url + 'contents/' + directory_path
    
    
def validate_onnx_file():
    pass
def validate_onnx_graph():
    with open("model.pkl", 'rb') as f:
        pkl = pickle.load(f)
    
def main():
    set_github_auth()
    parser = argparse.ArgumentParser(description='load onnx from github')
    parser.add_argument('--url', type=str,
                        help='github url which user provides')
    args = parser.parse_args()
    output = parse_github_url(args.url)
    print(output)

if __name__ == '__main__':
    main()
