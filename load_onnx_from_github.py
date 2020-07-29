import argparse
import json
import requests
import json
import tempfile
import os
import os.path as osp
import onnx

github_username = ""
github_token = ""

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
        print(e)
        
def download_from_directory(url, save_directory, exclude_name={}):
    """
    Download all files under the url dircetory recursively
    Will save all files into a corresponding path locally
    Args:
        url (str): GitHub url. 
        e.g., https://github.com/jcwchen/MLabHack2020/tree/master/data/onnx/mnist
        save_directory (str): save path locally
        (Optional) exclude_name (str): exclude some files or directories  
    """    
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
    """
    return onnx model path name.
    """
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.onnx'):
                return osp.join(path, filename)
    return None


def parse_github_url(url):
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
    # create a temp directory for downloading model; will remove if the validation is done
    with tempfile.TemporaryDirectory() as tmp:
        # download from model/
        model_directory_path = osp.join(tmp, 'model')
        os.mkdir(model_directory_path)
        download_from_directory(osp.join(content_url, 'model'), model_directory_path)
        onnx_path = find_onnx_file_in_directoy(model_directory_path) #.replace('\\', '/')
        print(onnx_path)
        try:
            # onnx model validation
            onnx.checker.check_model(onnx_path)
        except:
            # invalid onnx file
            return {'status': -1}
        
    # valid GitHub url with valid file 
    return {'status': 1, 'star_count': star_count, 'owner_name': owner_name}


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
    return result['stargazers_count'], result['owner']['login']


def get_github_json(url):
    """
    Get JSON with github token (to avoid API call limit) 
    """
    response = requests.get(url, auth=(github_username, github_token)).text
    return json.loads(response)

    
def main():
    parser = argparse.ArgumentParser(description='load onnx from github')
    parser.add_argument('--url', type=str,
                        help='github url which user provides')
    args = parser.parse_args()
    set_github_auth()
    output_json = parse_github_url(args.url)
    print(output_json)


if __name__ == '__main__':
    main()
