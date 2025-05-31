import requests
import zipfile
import sys
import os
import subprocess
import shutil

b2_version = '5.3.2'

zip_foldername = 'b2-' + b2_version
zip_filename = zip_foldername + '.zip'
b2_url = 'https://github.com/bfgroup/b2/releases/download/' + b2_version + '/b2-' + b2_version + '.zip'

def get_b2():
    response = requests.get(b2_url)
    with open(zip_filename, 'wb') as f:
        f.write(response.content)

    os.makedirs('.', exist_ok=True)

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('.')

    os.chdir(zip_foldername)
    subprocess.run([".\\bootstrap.bat"])
    os.chdir(r'..')

    b2_path = os.getcwd() + "\\b2"
    os.chdir(zip_foldername)
    subprocess.run(["b2", "install", "--prefix=" + b2_path])
    sys.path.append(b2_path)

    os.chdir(r'..')

    if os.path.isfile(zip_filename):
        os.remove(zip_filename)

    if os.path.isdir(zip_foldername):
        shutil.rmtree(zip_foldername)


def build_cpp():
    #b2 --toolset=msvc --build-type=complete architecture=x86 address-model=64 release debug
    os.chdir(r'siglib')
    subprocess.run(["b2", "--toolset=msvc", "--build-type=complete", "architecture=x86", "address-model=64", "release"])
    os.chdir(r'..')

