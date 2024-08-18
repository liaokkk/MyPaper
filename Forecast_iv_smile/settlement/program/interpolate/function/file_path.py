import numpy as np
def slash_tran(path):#檔案路徑字串可以允許不同的斜線(\ 或 / 都可以)，此程式會將他們統一轉成反斜線('\\')輸出
    dirs = path.split('/')
    path_new = dirs[0]
    for dir in dirs[1:]:
        path_new = path_new + '\\' + dir
    return path_new
def path_tran(old_path, top_path, new_dir):
    #top_path = './../dir1/dir2/'
    #old_path = './../dir1/dir2/dir3\\dir4\\\\file'
    #new_dir = 'New'
    #輸出為 '.\..\dir1\dir2\New\dir4\file'
    path = slash_tran(old_path)
    top_path = slash_tran(top_path)
    top_path_list = np.array(top_path.split('\\'))
    top_path_list = list(top_path_list[np.not_equal(top_path_list, '')])

    old_dir_idx = len(top_path_list)
    new_path_list = np.array(path.split('\\'))
    new_path_list = list(new_path_list[np.not_equal(new_path_list, '')])

    new_path_list[old_dir_idx] = new_dir
    new_path = new_path_list[0]
    for dir in new_path_list:
        new_path = new_path + '\\' + dir
    return new_path