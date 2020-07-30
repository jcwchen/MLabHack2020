# MLabHack2020

https://garagehackbox.azurewebsites.net/hackathons/2107/projects/93460

This repo validates the onnx model and get metadata from Github. 

# Installation

### Window
Install Python >= 3.6 
Install Python libraries to run `load_onnx_from_github.py`

```
install_py_lib.bat
```
### To increase calling API limit, you need to set Github Authentication
To do so, you need to manually create another python file `secret_config.py`. That file needs to contain:
```
GITHUB_USERNAME = 'YOUR_GITHUB_USERNAME'
GITHUB_TOKEN = 'YOUR_GITHUB_ACCESS_TOKEN'
```
You can obtain a GitHub Access Token from the GitHub Settings -> Developer settings -> Personal access tokens -> Generate new token -> check `read:packages` -> get the GitHub Access Token


# Run
Run `load_onnx_from_github.py` with arguments to output the result JSON.
```
python load_onnx_from_github.py --url [github_link]
# python load_onnx_from_github.py --url https://github.com/jcwchen/MLabHack2020/tree/master/data/onnx/mnist
# python load_onnx_from_github.py --url https://github.com/jcwchen/MLabHack2020/tree/master/data/onnx/mnist --module_name base_model --obj_name baselineCNN
```
### Arguments
--url: github url which user provides from form

--module_name: module name which user provides from form

--obj_name: object name which user provides from form

### Output
Output a JSON file including status, star count of that repo, the owner name of that repo, the module validity and the target onnx model size. 
For example:
```
{'status': 1, 'star_count': 1, 'owner_name': 'jcwchen', 'module_validity': True, 'onnx_size': 13104783}
```

### Status
As described in [load_onnx_from_github.py](load_onnx_from_github.py), the validation status is as follows:
```
class Status(Enum):
    SUCCESS = 1
    INVALID_ONNX = -1
    MULTIPLE_ONNX_FILES = -2
    INVALID_URL = -3
    ONNX_NOT_FOUND = -4
    MODULE_NOT_FOUND = -5
    INVALID_CLASS_NAME = -6
```
* SUCCESS (1) means parse the Github link successfully.
* INVALID_ONNX (-1) means the target onnx model is not valid which cannot pass `onnx.checker.check_model`
* MULTIPLE_ONNX_FILES (-2) means there are more than 1 onnx model in the target directoy. Current parser only allows single onnx model.
* INVALID_URL (-3) means the parser cannot parse the JSON from the GitHub url. The provided link might be broken.
* ONNX_NOT_FOUND (-4) means there are no `.onnx` model in the target directory.
* MODULE_NOT_FOUND (-5) means there are no `.py` module file defined in the target directory.
* INVALID_CLASS_NAME (-6) means the pytorch class/object name is invalid. Could be the case that the class is not a subclass of nn.Module, or 
the class is not defined in the '.py' file.

# Call python via C#
To call a python script via C#, you can create a function like:
```
@using System.Diagnostics
@using System.IO
@code {
    private String response = "";
    private void run_cmd()
    {
        String cmd = "C:\\your\\path\\to\\MLabHack2020\\load_onnx_from_github.py";
        String arg = "--url https://github.com/jcwchen/MLabHack2020/tree/master/data/onnx/mnist";
        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = "C:\\your\\path\\to\\Miniconda3\\python.exe";
        start.Arguments = string.Format("{0} {1}", cmd, arg);
        start.UseShellExecute = false;
        start.RedirectStandardOutput = true;
        using (Process process = Process.Start(start))
        {
            using (StreamReader reader = process.StandardOutput)
            {
                string result = reader.ReadToEnd();
                response = result;
                Console.Write(result);
            }
        }
    }
}
```
* Modify `cmd` to your `load_onnx_from_github.py` path.
* Modify `start.FileName` to your local installed python path.
* Modify `arg` to customize your arguments to the python script.
* Get `result` from the output of the target python script. (JSON)
* [Response.razor](Response.razor) is an example.

