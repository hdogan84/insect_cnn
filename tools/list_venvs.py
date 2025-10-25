import os

# Common venv folder names
venv_names = {"venv", ".venv", "env", ".env", "ENV"}

# Folders where venvs are often found
search_dirs = [
    os.path.expanduser("~"),             # your home directory
    os.path.join(os.getcwd()),           # current directory
    "C:\\Users\\dgnhk",                      # custom project directory (optional)
]

found_venvs = []

for root_dir in search_dirs:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) in venv_names and os.path.isdir(dirpath):
            python_exe = os.path.join(dirpath, "Scripts", "python.exe")
            if os.path.exists(python_exe):
                found_venvs.append(python_exe)

print("\n🧠 Found virtual environments:\n")
if found_venvs:
    for v in found_venvs:
        print(v)
else:
    print("No virtual environments found.")
