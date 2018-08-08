import argparse
import numpy as np


class Constants:
    Supporter = "Supporter"
    Answer = "Answer"
    Question = "Question"
    Context = "Context"


def get_args(flag):
    if flag == "train":
        parser = argparse.ArgumentParser()
        parser.add_argument("-b", "--task_id", help="specify babi task 1-20 (default=1)")
        parser.add_argument("-r", "--restore", help="restore previously trained weights (default=false)")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("-b", "--task_id", help="specify babi task 1-20 (default=1)")
    return parser.parse_args()


def get_task_name(id):
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",

    }

    return babi_map[id]


class Model_Configuration(object):
    """model hyperparams and data information."""

    batch_size = 100
    embed_size = 80
    hidden_size = 80

    max_epochs = 40
    terminate_epoch_rate = 10

    dropout = 0.9
    learning_rate = 0.001

    word2vec = True

    hops = 3

    max_allowed_inputs = 130
    train_split = 9000

    floatX = np.float32

    babi_id = "1"
    babi_test_id = ""

    train_mode = True
