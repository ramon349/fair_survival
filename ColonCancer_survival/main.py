import pandas as pd
import os
import argparse
import sys
import train

_data_root="/media/Datacenter_storage/Colon_pathology/DataCodebase/"
# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str,
                    default="./model/", 
                    help='path to the trained model')
parser.add_argument('--traindata', type=str, default=f"{_data_root}train.csv",
                    help='path to the train .csv file that contains comments with labels ')
parser.add_argument('--validationdata', type=str, default=f"{_data_root}internal_test.csv", 
                    help='path to the validation file .csv that contains comments with labels; only needed for training')
parser.add_argument('--testdata', type=str, default=f"{_data_root}external_test.csv", 
                    help='path to the test file .csv that contains comments without labels; only needed for testing ')
parser.add_argument('--savepath', type=str, default="./output/Test_prediction.csv", 
                    help='path to save the annotated test file .csv that contains model derieved prediction; only needed for testing')
parser.add_argument('--flag', type=str, default='Test',choices=['Train','Test'], 
                    help='flag to signify the mode of model use -  Train/Test')



# parse the arguments
args = parser.parse_args()


def main():
    surCox = train.Survial_Model()
    if args.flag == 'Train':
        try:
            traindf = pd.read_csv(args.traindata)
            validdf = pd.read_csv(args.validationdata)
            surCox.train_main(traindf, validdf)
            surCox.model_save(args.modelpath)
        except Exception as e :
            print(e)
            sys.exit('Enter the path for the correct .csv file or model saving path')
    if args.flag == 'Test':
        testdf = pd.read_csv(args.testdata)
        surCox.model_load(args.modelpath)
        annotated_test = surCox.test_main(testdf)
        print('finished the COX prediction')
        annotated_test.to_csv(args.savepath)
        print('Saved the data to: '+args.savepath)
      
    


if __name__ == "__main__":
    main()