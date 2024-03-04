import argparse
import json
from collections import (
    deque,
)  # just for fun using dequeue instead of just a list for faster appends
from pprint import pprint


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        data_dict = json.load(values)
        arg_list = deque()
        action_dict = {e.option_strings[0]: e for e in parser._actions}
        for i, e in enumerate(data_dict):
            arg_list.extend(self.__build_parse_arge__(e, data_dict, action_dict))
        parser.parse_args(arg_list, namespace=namespace)

    def __build_parse_arge__(self, arg_key, arg_dict, file_action):
        arg_name = f"--{arg_key}"
        arg_val = str(arg_dict[arg_key]).replace(
            "'", '"'
        )  # list of text need to be modified so they can be parsed properly
        try:
            file_action[arg_name].required = False
        except:
            raise KeyError(
                f"The Key {arg_name} is not an expected parameter. Delete it from config or update build_args method in helper_utils.configs.py"
            )
        return arg_name, arg_val


def parse_bool(s: str):
    return eval(s) == True


def warn_optuna(s: str):
    """Take string input of parser for optuna param and just output a warning
    Converts string to boolean for proper reading
    """
    val = eval(s)
    if val:
        print(val)
        print(
            f"WARNING YOU HAVE SELECTED OPTUNA PARAM SEARCH. Most PARAMS WILL BE IGNORED"
        )
    return val


def build_args():
    """Parses args. Must include all hyperparameters you want to tune.

    Special Note:
        Since i entirely expect to load from config files the behavior is
        1.
    """
    parser = argparse.ArgumentParser(
        description="Confguration for my deep learning model training for segmentation"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to pickle file containing tuple of form (train_set,val_set,test_set) see Readme for more)",
    ) 
    parser.add_argument(
        "--config_path", required=False, type=open, action=LoadFromFile, help="Path"
    )
    parser.add_argument("--epochs", required=True, type=int, help="")
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--batch_size", required=True, type=int) 
    parser.add_argument("--dataset",type=str,required=True,choices=['colon','mammo'],help="This is the nickname of the data used. Right now it's either colon or mammo")
    parser.add_argument('--subsets',type=json.loads,required=True,help='Defines the subsets used for model training,tuning,vlaidation and testing, for these experiemtns we also have an external dataset')
    parser.add_argument("--w_column",type=str,required=True,help="This will be the dataframe column that represents our proxy")
    parser.add_argument("--c_column",type=str,help="This defines our concept column",required=True)
    parser.add_argument("--patient_identifier",type=str,help='',required=True)
    parser.add_argument("--feature_names",type=json.loads,help='',required=True) 
    parser.add_argument("--model_name",type=str,required=True) 
    parser.add_argument("--learning_rate",type=float,required=True)
    parser.add_argument("--width",type=int,required=True)
    parser.add_argument("--latent_dim",type=int,required=True)
    parser.add_argument("--kl_coeff",type=int,required=True)
    parser.add_argument("--vae_epoch",type=int,required=True)
    parser.add_argument("--x2u_epoch",type=int,required=True)
    parser.add_argument("--xu2y_epoch",type=int,required=True)
    parser.add_argument("--calib_epoch",type=int,required=True)
    parser.add_argument("--run_param_search",type=bool,default=False,required=False)
    parser.add_argument("--seed",type=int,default=123) 
    parser.add_argument('--log_dir',type=str,required=True)
    parser.add_argument('--w_weights',type=json.loads,required=False) 
    parser.add_argument('--c_weights',type=json.loads,required=False)
    parser.add_argument("--enc_y_loss",type=str,choices=['classification','survival'],required=False)
    return parser

def build_test_args(): 
    parser = argparse.ArgumentParser(
        description="Confguration for my deep learning model testing for segmentation"
    )
    parser.add_argument('--model_weight',required=True) 
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--device',default='cuda:0',required=False)
    return parser
def build_infer_args(): 
    parser = argparse.ArgumentParser(
        description="Confguration to run inference of my model"
    )
    parser.add_argument('--config-path',required=False,type=open,action=LoadFromFile)
    parser.add_argument('--model_weight',required=False) 
    parser.add_argument('--pkl_path',required=False)
    parser.add_argument('--output_dir',required=False)
    parser.add_argument('--device',default='cuda:0',required=False)
    return parser

def get_params():
    args = build_args()
    my_args = args.parse_args()
    arg_dict = vars(my_args) 
    return arg_dict
def get_test_params(): 
    args = build_test_args() 
    my_args = args.parse_args()
    arg_dict = vars(my_args) 
    return arg_dict 
def get_infer_params():
    args = build_infer_args() 
    my_args = args.parse_args()
    arg_dict = vars(my_args) 
    return arg_dict 




if __name__ == "__main__":
    args = build_args()
    my_args = args.parse_args()
    arg_dict = vars(my_args)
    pprint(arg_dict)
