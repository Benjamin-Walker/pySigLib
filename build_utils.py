import zipfile
import subprocess
import traceback
import shutil
import os

import requests

B2_VERSION = '5.3.2'

ZIP_FOLDERNAME = 'b2-' + B2_VERSION
ZIP_FILENAME = ZIP_FOLDERNAME + '.zip'
B2_URL = 'https://github.com/bfgroup/b2/releases/download/' + B2_VERSION + '/b2-' + B2_VERSION + '.zip'

def _run(cmd, log_file, shell = False):
    try:
        output = subprocess.run(cmd, capture_output=True, check=True, text=True, shell = shell)
        log_file.write(output.stdout)
        log_file.write(output.stderr)
        return output
    except subprocess.CalledProcessError as e:
        log_file.write("\n" + "=" * 10 + " Exception occurred " + "=" * 10 + "\n")
        log_file.write("Exception occured whilst processing the command:\n\n")
        cmd_str = ""
        for c in cmd:
            cmd_str += c + " "
        log_file.write(cmd_str + "\n")
        log_file.write(repr(e.stdout))
        log_file.write(repr(e.stderr))
        raise e
    except Exception as e:
        log_file.write("\n" + "=" * 10 + " Exception occurred " + "=" * 10 + "\n")
        log_file.write("Exception occured whilst processing the command:\n\n")
        cmd_str = ""
        for c in cmd:
            cmd_str += c + " "
        log_file.write(cmd_str + "\n")
        traceback.print_exc(file=log_file)
        raise e

def get_paths(system, log_file):
    if 'CUDA_PATH' not in os.environ:
        raise RuntimeError("Error while compiling pysiglib: CUDA_PATH environment variable not set")

    cuda_path = os.environ['CUDA_PATH']

    vctoolsinstalldir = get_msvc_path(log_file)
    cl_path = os.path.join(vctoolsinstalldir, 'bin', 'HostX64', 'x64')
    os.environ["PATH"] += os.pathsep + cl_path

    idx = vctoolsinstalldir.find("VC")
    path = vctoolsinstalldir[:idx]

    output = _run([os.path.join(path, 'Common7', 'Tools', 'VsDevCmd.bat'), '&&', 'set'], log_file, shell = True)

    log_file.write(output.stdout + output.stderr)
    output = output.stdout
    start = output.find('INCLUDE') + 8
    end = output[start:].find('\n')
    include = output[start: start + end]

    dir_ = os.getcwd()
    return dir_, vctoolsinstalldir, cl_path, cuda_path, include

def get_b2(system, log_file):
    response = requests.get(B2_URL, timeout=(5, 60), stream=True)
    with open(ZIP_FILENAME, 'wb') as f:
        f.write(response.content)

    os.makedirs('.', exist_ok=True)

    with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall('.')

    os.chdir(ZIP_FOLDERNAME)
    if system == 'Windows':
        _run([".\\bootstrap.bat"], log_file)
    elif system == 'Linux':
        raise #TODO
    elif system == 'Darwin':
        _run(["chmod", "-R", "755", "."], log_file)
        _run(["./bootstrap.sh"], log_file)
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + system + "' in get_b2()")

    os.chdir(r'..')


    os.chdir(ZIP_FOLDERNAME)
    _run(["./b2", "install", "--prefix=../b2"], log_file)
    os.chdir(r'..')
    # b2_path = os.getcwd() + "\\b2\\bin"
    # sys.path.append(b2_path)

    # os.chdir(r'..')

    if os.path.isfile(ZIP_FILENAME):
        os.remove(ZIP_FILENAME)

    if os.path.isdir(ZIP_FOLDERNAME):
        shutil.rmtree(ZIP_FOLDERNAME)


def build_cpsig(system, log_file):
    os.chdir(r'siglib')
    if system == 'Windows':
        _run(["../b2/b2", "--toolset=msvc", "--build-type=complete", "architecture=x86", "address-model=64", "release"], log_file)
    elif system == 'Linux':
        raise #TODO
    elif system == 'Darwin':
        _run(["chmod", "755", "../b2"], log_file)
        _run(["../b2/bin/b2", "--build-type=complete", "release"], log_file)
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + system + "' in build_cpsig()")
    os.chdir(r'..')

def build_cusig(system, log_file):
    _, vctoolsinstalldir, _, _, _ = get_paths(system, log_file)
    vc0 = vctoolsinstalldir[:vctoolsinstalldir.find(r'\Tools')]
    if system == 'Windows':
        _run(["C:\\Users\\Shmelev\\source\\repos\\pySigLib\\build_cusig.bat", vc0, vctoolsinstalldir], log_file)
    elif system == 'Linux':
        raise #TODO
    elif system == 'Darwin':
        raise #TODO
    else:
        # Shouldn't really end up here, but just in case
        raise RuntimeError("Unknown error while building pysiglib: unexpected system '" + system + "' in build_cusig()")

def get_msvc_path(log_file):
    os.chdir('siglib')
    output = _run(["b2", "toolset=msvc", "--debug-configuration", "-n"], log_file)
    os.chdir('..')
    output = output.stdout
    log_file.write(output + "\n")
    idx = output.find("[msvc-cfg] msvc-")
    output = output[idx:]
    start = output.find("'") + 1
    end = output.find("bin") - 1

    if idx == -1 or start == 0 or end == -2:
        raise RuntimeError("Error while compiling pysiglib: MSVC not found")

    return output[start: end]

def nvcc_compile_and_link(files, dir_, cl_path, cuda_path, include):

    for filename in files:
        nvcc_compile_file_(filename, dir_, cl_path, cuda_path, include)

    nvcc_link(files, dir_, cuda_path)


def nvcc_compile_file_(filename, dir_, cl_path, cuda_path, include):

    commands = [
        os.path.join(cuda_path, 'bin', 'nvcc.exe'),
        '-gencode=arch=compute_52,code=\"sm_52,compute_52\"',
        '--use-local-env',
        '-ccbin', cl_path,
        '-x', 'cu', '-rdc=true',
        # f'-I{CUDA_PATH}\\include',
        # f'-I{VCTOOLSINSTALLDIR}\\include'
    ]

    commands += ['-I' + x for x in include.split(';')]

    commands += [
        '-diag-suppress', '108',
        '-diag-suppress', '174',
        '--keep-dir_', 'x64\\Release',
        '-maxrregcount=0',
        '--machine', '64',
        '--compile', '-cudart', 'static', '-lineinfo',
        '-DNDEBUG', '-DCUSIG_EXPORTS', '-D_WINDOWS', '-D_USRDLL', '-D_WINDLL',
        '-D_UNICODE', '-DUNICODE',
        '-Xcompiler', '"/EHsc /W3 /nologo /O2 /FS /MT"',
        '-Xcompiler', '/Fdx64\\Release\\vc143.pdb',
        '-o', f'{dir_}\\siglib\\cusig\\{filename}.obj',
        f'{dir_}\\siglib\\cusig\\{filename}'
    ]

    subprocess.run(commands, check=True)

def nvcc_link(files, dir_, cuda_path):

    print(os.path.join(cuda_path, 'bin', 'crt'), "<--")
    print(os.path.join(cuda_path, 'lib', 'x64'))

    commands = [
        os.path.join(cuda_path, 'bin', 'nvcc.exe'),
        # '--verbose',
        '-dlink',
        '-o', 'siglib\\cusig\\cusig.device-link.obj',
        '-Xcompiler', '"/EHsc /W3 /nologo /O2 /MT"',
        # '-Xcompiler' '/Fdx64\Release/vc143.pdb',
        '-L' + os.path.join(cuda_path, 'bin', 'crt'),
        '-L' + os.path.join(cuda_path, 'lib', 'x64'),
        'kernel32.lib', 'user32.lib', 'gdi32.lib', 'winspool.lib',
        'comdlg32.lib', 'advapi32.lib', 'shell32.lib', 'ole32.lib',
        'oleaut32.lib', 'uuid.lib', 'odbc32.lib', 'odbccp32.lib',
        'cudart.lib', 'cudadevrt.lib',
        '-gencode=arch=compute_52,code=sm_52'
    ]
    commands += [f'{dir_}\\siglib\\cusig\\{filename}.obj' for filename in files]

    subprocess.run(commands, check=True)
