from task1 import task1
from task2 import task2
from task3 import task3

if __name__ == '__main__':
    meta_file_path = 'data/objects/book/meta.npy'
    single_shot = False
    single_shot_number = 0
    visualize = True

    print("########## Task 1 ##########")
    task1(meta_file_path, single_shot, single_shot_number, visualize)

    print("\n\n########## Task 2 ##########")
    task2(meta_file_path, single_shot_number, visualize)

    print("\n\n########## Task 3 ##########")
    task3(meta_file_path, single_shot, single_shot_number, visualize)

    print("\n\n########## Done ##########")