import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import optuna as opt 
import tensorflow as tf
import pickle as pkl 
from fair_survival.helper_utils.colon_cancer_proc import processor as colon_data_processor
import json 
import pandas as pd 
import tensorflow as tf
import pdb 
import pickle as pkl 
from .helper_utils.data_proc import my_ret_func  ,make_set_by_split,get_x_only
from .model_dist import model_factory 
from .helper_utils import configs as  help_configs 
import numpy as np  
from pathlib import Path 
import os 
def figure_version(log_dir): 
    logs = [i for i,e in enumerate(Path(log_dir).glob('run_*'))] 
    if len(logs)>0: 
        ver = logs[-1] +1
    else: 
        ver = 0 
    log_path = os.path.join(log_dir,f'run_{ver}')
    os.makedirs(log_path,exist_ok=True)
    return log_path

def  is_survivor(*args,key_val=0): 
    val_com = args[-1]==key_val
    return val_com
def main(train_config): 
    SEED = train_config['seed']
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    #load the data and define the feature values 
    colon_df = pd.read_csv(train_config['data_path']) 
    log_path = figure_version(train_config['log_dir'])
    FEATURE_NAMES=train_config['feature_names'] 
    input_dict = {}
    subset_d = train_config['subsets']
    for k in subset_d: 
        input_dict[k] = make_set_by_split(colon_df,subset_d[k],FEATURE_NAMES,config=train_config)
    batch_size = train_config['batch_size']
    ds_dict_source = {
    key: tf.data.Dataset.from_tensor_slices(
            (value['x'], value['y_one_hot'], value['c'], value['w_one_hot'],value['time_to_event'],value['event']), 
        ).repeat().shuffle(batch_size*4).batch(batch_size) for key, value in input_dict.items() #TODO: I should update this shuffling stuff
    } 

    test_ds_dict_source = {
    key: tf.data.Dataset.from_tensor_slices(
            (value['x'],value['time_to_event'],value['event']), 
        ).batch(batch_size) for key, value in input_dict.items()

    }
    train_set = ds_dict_source['train'].map(my_ret_func)
    num_examples =  (colon_df['split']=='internal-train').sum()
    steps_per_epoch_train = num_examples // batch_size
    steps_per_epoch_val =   (colon_df['split']=='internal-val').sum()//batch_size

    model_name = train_config['model_name']
    model= model_factory(model_name=model_name,config=train_config,sample_data=input_dict['train']) 
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', min_delta=0.01, factor=0.1, patience=20,
    min_lr=1e-12)
    callbacks = [reduce_lr]
    if model_name=='causal': 
        train_kwargs = {
        'steps_per_epoch':steps_per_epoch_train,
        'stpes_per_epoch_val':steps_per_epoch_val,
        'verbose': True,
        'callbacks':callbacks,
        'vae_epoch':train_config['vae_epoch'],
        'x2u_epoch':train_config['x2u_epoch'],
        'xu2y_epoch':train_config['xu2y_epoch'],
        'calib_epoch':train_config['calib_epoch'],
            }
        model.fit(ds_dict_source['train'], ds_dict_source['val'], ds_dict_source['tune'],
            steps_per_epoch_val, **train_kwargs)
        histories = pd.DataFrame(model.vae.history.history) 
        history_path = os.path.join(log_path,'causal_model_history.csv')
        histories.to_csv(history_path,index=False)
    else: 
        train_kwargs = {
        'epochs':train_config['epochs'],
        'callbacks':callbacks,
        'steps_per_epoch':steps_per_epoch_train,
        'verbose':True,
        }
        model.fit(train_set,**train_kwargs,callbacks=None)
    if model_name=='causal': 
        split_dfs = list() 
        for split in test_ds_dict_source.keys(): 
            predictions = list() 
            for encode_in in iter(test_ds_dict_source[split].map(get_x_only)):
                hazard = model.predict_hazard(encode_in)
                predictions.append(hazard)
            all_preds= np.vstack(predictions)
            for k in range(all_preds.shape[1]): 
                input_dict[split][f'risk_{k}']= all_preds[:,k] 
            split_df = pred_dict_to_frame(input_dict[split],train_config) 
            split_df['split'] = split
            split_dfs.append(split_df)
        test_df = pd.concat(split_dfs)
        csv_path = os.path.join(log_path,'model_infer.csv') 
        test_df.to_csv(csv_path,index=False) 
    config_save_path = os.path.join(log_path,'orig_config.json') 
    with open(config_save_path,'w') as f : 
        json.dump(train_config,f)
def pred_dict_to_frame(pred_dict,config): 
    pid = config['patient_identifier'] 
    cols = [pid]
    cols = cols + [e for e in pred_dict.keys() if e.startswith('risk_')]
    cols2copy = {e:pred_dict[e].reshape((-1,)) for e in cols}
    copy_df = pd.DataFrame(cols2copy) 
    return copy_df
def _parse(): 
    conf = help_configs.get_params() 
    if conf['run_param_search']: 
        raise NotImplemented("Working on it")
    else: 
        main(conf)
if __name__=='__main__':
    _parse()

