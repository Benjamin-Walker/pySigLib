import requests
import zipfile
import sys
import subprocess
import shutil
import os

b2_version = '5.3.2'

zip_foldername = 'b2-' + b2_version
zip_filename = zip_foldername + '.zip'
b2_url = 'https://github.com/bfgroup/b2/releases/download/' + b2_version + '/b2-' + b2_version + '.zip'

def get_paths(SYSTEM):
    if 'CUDA_PATH' not in os.environ:
        raise RuntimeError("Error while compiling pysiglib: CUDA_PATH environment variable not set")

    CUDA_PATH = os.environ['CUDA_PATH']

    VCTOOLSINSTALLDIR = get_msvc_path()
    CL_PATH = os.path.join(VCTOOLSINSTALLDIR, 'bin', 'HostX64', 'x64')
    os.environ["PATH"] += os.pathsep + CL_PATH

    idx = VCTOOLSINSTALLDIR.find("VC")
    path = VCTOOLSINSTALLDIR[:idx]

    output = subprocess.run([os.path.join(path, 'Common7', 'Tools', 'VsDevCmd.bat'), '&&', 'set'], capture_output=True,
                            text=True, shell=True)
    output = output.stdout
    start = output.find('INCLUDE') + 8
    end = output[start:].find('\n')
    INCLUDE = output[start: start + end]

    DIR = os.getcwd()
    return DIR, VCTOOLSINSTALLDIR, CL_PATH, CUDA_PATH, INCLUDE

def get_b2(SYSTEM):
    response = requests.get(b2_url)
    with open(zip_filename, 'wb') as f:
        f.write(response.content)

    os.makedirs('.', exist_ok=True)

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('.')

    os.chdir(zip_foldername)
    if SYSTEM == 'Windows':
        subprocess.run([".\\bootstrap.bat"])
    elif SYSTEM == 'Linux':
        raise #TODO
    elif SYSTEM == 'Darwin':
        subprocess.run(["chmod", "-R", "755", "."])
        subprocess.run(["./bootstrap.sh"])
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + SYSTEM + "' in get_b2()")

    os.chdir(r'..')


    os.chdir(zip_foldername)
    subprocess.run(["./b2", "install", "--prefix=../b2"])
    os.chdir(r'..')
    # b2_path = os.getcwd() + "\\b2\\bin"
    # sys.path.append(b2_path)

    # os.chdir(r'..')

    if os.path.isfile(zip_filename):
        os.remove(zip_filename)

    if os.path.isdir(zip_foldername):
        shutil.rmtree(zip_foldername)


def build_cpsig(SYSTEM):
    os.chdir(r'siglib')
    if SYSTEM == 'Windows':
        subprocess.run(["../b2/b2", "--toolset=msvc", "--build-type=complete", "architecture=x86", "address-model=64", "release"])
    elif SYSTEM == 'Linux':
        raise #TODO
    elif SYSTEM == 'Darwin':
        subprocess.run(["chmod", "755", "../b2"])
        subprocess.run(
            ["../b2/bin/b2", "--build-type=complete", "release"])
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + SYSTEM + "' in build_cpsig()")
    os.chdir(r'..')

def build_cusig(SYSTEM):
    DIR, VCTOOLSINSTALLDIR, CL_PATH, CUDA_PATH, INCLUDE = get_paths(SYSTEM)
    VC0 = VCTOOLSINSTALLDIR[:VCTOOLSINSTALLDIR.find(r'\Tools')]
    if SYSTEM == 'Windows':
        subprocess.run(["C:\\Users\Shmelev\source\\repos\pySigLib\\build_cusig.bat", VC0, VCTOOLSINSTALLDIR])
    elif SYSTEM == 'Linux':
        raise #TODO
    elif SYSTEM == 'Darwin':
        raise #TODO
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + SYSTEM + "' in build_cusig()")

def get_msvc_path():
    output = subprocess.run(["b2", "toolset=msvc", "--debug-configuration"], capture_output=True, text=True)
    output = output.stdout
    idx = output.find("[msvc-cfg] msvc-")
    output = output[idx:]
    start = output.find("'") + 1
    end = output.find("bin") - 1

    if idx == -1 or start == 0 or end == -2:
        raise RuntimeError("Error while compiling pysiglib: MSVC not found")

    return output[start: end]

def nvcc_compile_and_link(files, DIR, CL_PATH, CUDA_PATH, INCLUDE):

    for filename in files:
        nvcc_compile_file_(filename, DIR, CL_PATH, CUDA_PATH, INCLUDE)

    nvcc_link(files, DIR, CUDA_PATH)


def nvcc_compile_file_(filename, DIR, CL_PATH, CUDA_PATH, INCLUDE):

    commands = [
        os.path.join(CUDA_PATH, 'bin', 'nvcc.exe'),
        '-gencode=arch=compute_52,code=\"sm_52,compute_52\"',
        '--use-local-env',
        '-ccbin', CL_PATH,
        '-x', 'cu', '-rdc=true',
        # f'-I{CUDA_PATH}\\include',
        # f'-I{VCTOOLSINSTALLDIR}\\include'
    ]

    commands += ['-I' + x for x in INCLUDE.split(';')]

    commands += [
        '-diag-suppress', '108',
        '-diag-suppress', '174',
        '--keep-dir', 'x64\\Release',
        '-maxrregcount=0',
        '--machine', '64',
        '--compile', '-cudart', 'static', '-lineinfo',
        '-DNDEBUG', '-DCUSIG_EXPORTS', '-D_WINDOWS', '-D_USRDLL', '-D_WINDLL',
        '-D_UNICODE', '-DUNICODE',
        '-Xcompiler', '"/EHsc /W3 /nologo /O2 /FS /MT"',
        '-Xcompiler', '/Fdx64\\Release\\vc143.pdb',
        '-o', f'{DIR}\\siglib\\cusig\\{filename}.obj',
        f'{DIR}\\siglib\\cusig\\{filename}'
    ]

    subprocess.run(commands)

def nvcc_link(files, DIR, CUDA_PATH):

    print(os.path.join(CUDA_PATH, 'bin', 'crt'), "<--")
    print(os.path.join(CUDA_PATH, 'lib', 'x64'))

    commands = [
        os.path.join(CUDA_PATH, 'bin', 'nvcc.exe'),
        # '--verbose',
        '-dlink',
        '-o', 'siglib\\cusig\\cusig.device-link.obj',
        '-Xcompiler', '"/EHsc /W3 /nologo /O2 /MT"',
        # '-Xcompiler' '/Fdx64\Release/vc143.pdb',
        '-L' + os.path.join(CUDA_PATH, 'bin', 'crt'),
        '-L' + os.path.join(CUDA_PATH, 'lib', 'x64'),
        'kernel32.lib', 'user32.lib', 'gdi32.lib', 'winspool.lib',
        'comdlg32.lib', 'advapi32.lib', 'shell32.lib', 'ole32.lib',
        'oleaut32.lib', 'uuid.lib', 'odbc32.lib', 'odbccp32.lib',
        'cudart.lib', 'cudadevrt.lib',
        '-gencode=arch=compute_52,code=sm_52'
    ]
    commands += [f'{DIR}\\siglib\\cusig\\{filename}.obj' for filename in files]

    subprocess.run(commands)