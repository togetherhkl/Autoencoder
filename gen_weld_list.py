"""
文档说明：
创建训练集和测试集的txt文件路劲，每个文件夹的前十张图片为测试集，其余为训练集
"""
import os
def gen_weld_list(name, path):
    """
    功能：创建训练集和测试集
    参数：
        name：训练集或测试集名称
        path：总数据集路径
    """
    folders = os.listdir(path)

    if not os.path.exists("list"):
        os.makedirs("list")

    fl_train = open('list/' + name + '_train_list.txt', 'w')#训练集的txt文件
    fl_test = open('list/' + name + '_test_list.txt', 'w')#测试集的txt文件
    fn = open('list/' + name + '_project_name.txt', 'w')#项目名称文件
    fw = open('list/' + name + '_weld_num.txt', 'w')#底片数量文件
    fi = open('list/' + name + '_info.txt', 'w')#项目信息文件

    #统计项目数量
    project_num = len(folders)

    #焊口总数
    total_weld_num = 0
    #切片总数
    total_patch_num = 0

    for i, folder in enumerate(folders):
            
            fn.write(folder + '\n')#将项目名称写入文件
    
            folder_path = os.path.join(path, folder)
            total_weld_num += len(os.listdir(folder_path))            

            subfoders = os.listdir(folder_path)
            for subfoder in subfoders:
                fw.write(folder + '/' + subfoder + '\n')#将底片名称写入文件
                files = os.listdir(os.path.join(folder_path, subfoder))
                total_patch_num += len(files)
                file_path = os.path.join(folder_path, subfoder)

                #前十张图片为测试集，其余为训练集
                for j, file in enumerate(files):
                    if j < 10:
                        fl_test.write('{}={}\n'.format(os.path.join(file_path, file), i))
                    else:
                        fl_train.write('{}={}\n'.format(os.path.join(file_path, file), i))
    
    fl_train.close()
    fl_test.close()
    fn.close()

    #计算训练集和测试集的数量
    with open('list/' + name + '_train_list.txt', 'r') as f:
        train_lines = f.readlines()
        train_num = len(train_lines)
    with open('list/' + name + '_test_list.txt', 'r') as f:
        test_lines = f.readlines()
        test_num = len(test_lines)

    #将项目信息写入文件
    fi.write('项目数量：{}\n'.format(project_num))
    fi.write('底片总数：{}\n'.format(total_weld_num))
    fi.write('切片总数：{}\n'.format(total_patch_num))
    fi.write('训练集数量：{}\n'.format(train_num))
    fi.write('测试集数量：{}\n'.format(test_num))
    fi.close()

    
    

if __name__ == '__main__':
    gen_weld_list('weld1016', '../../../Data/1016/train_weld_edge')#创建训练集和测试集
    print("训练集和测试集txt文件创建成功！")
            
    