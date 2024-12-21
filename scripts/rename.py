import os

def rename_files_in_folder(folder_path, file_extension):
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]

    files.sort()

    for i, filename in enumerate(files):
        new_name = f"{i + 1}{file_extension}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed {src} to {dst}")

def create_files(folder_path):
    for i in range(100,112):
        files = [f for f in os.listdir(folder_path) if f.endswith('txt')]
        if str(i)+'.txt' in files:
            continue
        f = open(f'{folder_path}\{i}.txt','w')
        f.close()

#rename_files_in_folder(folder_path = r'/content/lpg_dataset', file_extension='.jpg')