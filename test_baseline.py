import argparse

import baseline


def main():
    print("in test baseline")
    new_parser = argparse.ArgumentParser()
    new_parser.add_argument("-id", help="specify babi task id 1-20 (default is 1)")
    new_parser.add_argument("-sen", type=str, required=True)
    new_parser.add_argument("-que", type=str, required=True)
    test_args = new_parser.parse_args()
    task_id = test_args.id
    sen = ""
    q = ""
    if test_args.sen is not None:
        sen = test_args.sen
    if test_args.que is not None:
        q = test_args.que

    model_dir = "./baseline/model/model_{}.json".format(task_id)
    weight_dir = "./baseline/weight/model_{}.h5".format(task_id)
    input_file = "./baseline/inputs/input_{}.txt".format(task_id)
    baseline.test(model_dir, weight_dir,input_file, sen=sen, q=q)


if __name__ == '__main__': main()
