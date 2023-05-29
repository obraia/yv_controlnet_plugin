import os
import sys
import subprocess

plugin_dir = os.path.dirname(__file__)
controlnet_dir = os.path.join(plugin_dir, 'controlnet')
weights_dir = os.path.join(controlnet_dir, 'weights')

def setup():
    install_requirements()
    append_python_paths()

def install_requirements():
    requirements_path = os.path.join(plugin_dir, 'requirements.txt')
    out = subprocess.check_output(['pip', 'install', '-r', requirements_path])

    for line in out.splitlines():
        print(line)
        
def append_python_paths():
    if plugin_dir not in sys.path:
        sys.path.append(plugin_dir)

    if controlnet_dir not in sys.path:
        sys.path.append(controlnet_dir)
