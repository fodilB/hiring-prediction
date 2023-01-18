__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'


if __name__ == '__main__':
    # initializing list
    test_list = [4, 5, 8, 9, 10, 17]

    # printing list
    print("The original list : " + str(test_list))

    # Median of list
    # Using loop + "~" operator
    l = test_list[0:6]
    print(l, l[-1])
    test_list.sort()
    mid = len(test_list) // 2
    print(mid,test_list[mid],  ~mid,test_list[-1])
    res = (test_list[mid] + test_list[~mid]) / 2

    # Printing result
    print("Median of list is : " + str(res))