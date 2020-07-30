import argparse
import json
import requests
import tempfile
import os
import os.path as osp
import onnx
import sys
import importlib
import torch.nn as nn
from enum import Enum

class Status(Enum):
    SUCCESS = 1
    INVALID_ONNX = -1
    MULTIPLE_ONNX_FILES = -2
    INVALID_URL = -3
    ONNX_NOT_FOUND = -4
    MODULE_NOT_FOUND = -5
    INVALID_CLASS_NAME = -6

github_username = ""
github_token = ""
target_onnx_size = [0]
result_json = {'status': Status.SUCCESS.value}

def download_onnx_model(previous_result, deeper_download_url):
    result = get_github_json(deeper_download_url)
    # If file >= 100MB, it needs to be download via deeper download_url (GitHub LFS) 
    if 'download_url' in result:
        print(result['download_url'])
        target_onnx_size[0] = result['size']
        return result['download_url']
    # Otherwise, simply use download_url
    else:
        target_onnx_size[0] = previous_result['size']
        return previous_result['download_url']


def set_github_auth():
    """
    set GitHub authentication to increase API call limit
    Need a secret_config which includes GITHUB_USERNAME and GITHUB_TOKEN 
    """    
    try:
        import secret_config
        github_username = secret_config.GITHUB_USERNAME
        github_token = secret_config.GITHUB_TOKEN
    except Exception as e:
        pass
        # print('Warning: set GitHub authentication to increase API call limit.')
        # print(e)
        
def download_from_directory(url, save_directory, exclude_name={}):
    """
    Download all files under the url dircetory recursively
    Will save all files into a corresponding path locally
    Args:
        url (str): GitHub url. 
        e.g., https://github.com/jcwchen/MLabHack2020/tree/master/data/onnx/mnist
        save_directory (str): save path locally
        (Optional) exclude_name (str): exclude some files or directories
    Returns:
        Status: download status
    """
    result = get_github_json(url)
    if result_json['status'] != Status.SUCCESS.value:
        return result_json['status']
    for files in result:
        if files['name'].endswith('.onnx'):
            # multiple onnx files in the same model directory
            if target_onnx_size[0] != 0:
                return Status.MULTIPLE_ONNX_FILES.value
        if files['download_url']:
            download_url = download_onnx_model(files, files['url'])
            file_url = requests.get(download_url)
            file_path = osp.join(save_directory, files['name'])
            with open(file_path, 'wb') as f:
                f.write(file_url.content)
        elif files['type'] == 'dir' and files['name'] not in exclude_name:
            directory_path = osp.join(save_directory, files['name'])
            os.mkdir(directory_path)
            download_status = download_from_directory(files['url'].replace('?ref=master', ''), directory_path, exclude_name)
            if download_status != Status.SUCCESS.value:
                return download_status
    return Status.SUCCESS.value


def find_onnx_file_in_directoy(path):
    """
    Returns:
        str: onnx model path name
    """
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.onnx'):
                return osp.join(path, filename)
    return None

def find_module_file_in_directoy(path, module_fname):
    """
    return nn.module python file path name.
    """
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename == (module_fname + '.py'):
                return path
    return None

def validate_module_file(path, module_fname, class_name):
    sys.path.append(path)
    try:
        net = importlib.import_module(module_fname).__getattribute__(class_name)
    except:
        # print('Module not found')
        return False
    if not issubclass(net, nn.Module):
        # print('Module not a valid class')
        return False
    # print('success')
    return True

def parse_github_url(url, module_fname, class_name):
    """
    parse files, validate the target model and get metadata from the GitHub url
    Args:
        url (str): GitHub url. 
        e.g., https://github.com/jcwchen/MLabHack2020/tree/master/data/onnx/mnist
    Returns:
        dict: {validation status: int, star count: int, owner name: str} 
    """    
    api_url, content_url = convert2github_api(url)
    star_count, owner_name = get_github_metadata(api_url)
    if star_count is None:
        return result_json        
    # create a temp directory for downloading model; will remove if the validation is done
    with tempfile.TemporaryDirectory() as tmp:
        # download from model/
        model_directory_path = osp.join(tmp, 'model')
        os.mkdir(model_directory_path)
        download_status = download_from_directory(osp.join(content_url, 'model'), model_directory_path)
        # download fail
        if download_status != Status.SUCCESS.value:
            result_json['status'] = download_status
            return result_json
        # onnx model not found
        if target_onnx_size[0] == 0:
            result_json['status'] = Status.ONNX_NOT_FOUND.value
            return result_json
        onnx_path = find_onnx_file_in_directoy(model_directory_path) #.replace('\\', '/')
        module_path = find_module_file_in_directoy(model_directory_path, module_fname)
        # print('module path:' , module_path)
        # print('onnx path: ', onnx_path)
        module_valid = False
        if module_path:
            module_valid = validate_module_file(module_path, module_fname, class_name)
            if not module_valid:
                result_json['status'] = Status.INVALID_CLASS_NAME.value
        else:
            result_json['status'] = Status.MODULE_NOT_FOUND.value
        try:
            # onnx model validation
            onnx.checker.check_model(onnx_path)
        except:
            # invalid onnx file
            result_json['status'] = Status.INVALID_ONNX.value
    # load metadata from github
    result_json['star_count'] = star_count
    result_json['owner_name'] = owner_name
    result_json['module_validity'] = module_valid
    result_json['onnx_size'] = target_onnx_size[0]
    # default status is 1 if success

    return result_json


def convert2github_api(url):
    """
    Convert original GitHub path into GitHub API to get data 
    Args:
        url (str): GitHub url. 
        e.g., https://github.com/jcwchen/MLabHack2020/tree/master/data/onnx/mnist
    Returns:
        str: GitHub API url.
        e.g., https://api.github.com/repos/jcwchen/MLabHack2020/contents
    """
    api_url = 'https://api.github.com/repos'
    directory_path = ''
    i = -1
    for token in url.split('/'):
        i += 1
        # 0-2: https:// (useless)
        # 3-4: user name + repo name
        if 3 <= i <= 4:
            api_url += '/'+ token
        # 4-5: branch tree (useless) 
        # >=7: directory path
        elif i >= 7:
            directory_path += token + '/'
    return api_url, api_url + '/contents/' + directory_path
    

def get_github_metadata(url):
    """
    Get repo meta data
    """
    result = get_github_json(url)
    if result is None:
        return None
    else:
        return result['stargazers_count'], result['owner']['login']


def get_github_json(url):
    """
    Get JSON with github token (to avoid API call limit) 
    """
    result = None
    try:
        response = requests.get(url, auth=(github_username, github_token)).text
        result = json.loads(response)         
    except Exception as e:
        result_json['status'] = Status.INVALID_URL.value
        # print('Get Github API JSON fail from the link: {}'.format(url))
        # print(e)    
    return result

    
def main():
    parser = argparse.ArgumentParser(description='load onnx from github')
    parser.add_argument('--url', type=str,
                        help='github url which user provides')

    # for nn.Module validation
    parser.add_argument('--module_name', action='store', default='model', dest='module_fname', type=str)
    parser.add_argument('--obj_name', action='store', default='Net', dest='class_name', type=str)

    args = parser.parse_args()
    set_github_auth()
    output_json = parse_github_url(args.url, args.module_fname, args.class_name)
    # output JSON to C# server to catch
    print(output_json)


if __name__ == '__main__':
    main()
