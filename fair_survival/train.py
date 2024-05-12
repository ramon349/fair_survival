import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
from fair_survival.helper_utils.colon_cancer_proc import processor as colon_data_processor
import json 
import pandas as pd 
from .helper_utils.data_proc import my_ret_func  ,make_set_by_split,get_x_only,make_set_by_hospital
from .model_dist import model_factory 
from .helper_utils import configs as  help_configs 
import numpy as np  
import os 
from .helper_utils.data_proc import calculateBaselineHazard 
import matplotlib.pyplot as plt 
import pdb 
import optuna  as opt 
def figure_version(log_dir): 
    ver = 0 
    log_path = os.path.join(log_dir,f'run_{ver}')
    os.makedirs(log_path,exist_ok=True)
    return log_path

def  is_survivor(*args,key_val=0): 
    val_com = args[-1]==key_val
    return val_com
def build_optuna_params(param,trial): 
    param['batch_size']= trial.suggest_int("batch_size",2,512,2)
    param['learning_rate'] = trial.suggest_float("learning_rate",0.0001,0.1)
    param['hidden_dim'] = trial.suggest_int("hidden_dim",4,12,1) 
    return param
def sch(epoch,lr): 
    if epoch < 75: 
        return lr 
    else: 
        return lr* np.exp(-0.1)
def main(in_config,trial=None): 

    if trial: 
        #do stuff 
        train_config = build_optuna_params(in_config.copy(),trial)
        verbose =0 
    else: 
        train_config = in_config
        verbose = 1

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

    #creating dsets  
    ds_dict_source = dict() 
    for key,value in input_dict.items():
        if key=='val':
            m_batch = 124
        else: 
            m_batch = batch_size
        ds_dict_source[key] = tf.data.Dataset.from_tensor_slices((value['x'],value['y_one_hot'],value['c'],value['w_one_hot'],value['time_to_event'],value['event'])).repeat().shuffle(m_batch*4).batch(m_batch)

    test_ds_dict_source = {
    key: tf.data.Dataset.from_tensor_slices(
            (value['x'],value['time_to_event'],value['event']), 
        ).batch(batch_size) for key, value in input_dict.items()

    }
    centers = ['Ontario', 'Australia', 'Hawaii', 'Seattle', 'Mayo', 'UPMC','ACCESS', 'Mt. Sinai']
    center_ds_dict_source = dict() 
    for c in centers: 
        center_ds_dict_source[c] = make_set_by_hospital(colon_df,c,feat_names=FEATURE_NAMES) 
    center_ts_ds = {
        key:tf.data.Dataset.from_tensor_slices(

            (value['x'],value['y_one_hot'],value['c'],value['w_one_hot'],value['time_to_event'],value['event'])).batch(batch_size) for key,value in input_dict.items()}
    train_set = ds_dict_source['train'].map(my_ret_func)
    num_examples =  (colon_df['split']=='internal-train').sum()
    steps_per_epoch_train = num_examples // batch_size
    steps_per_epoch_val =   (colon_df['split']=='internal-val').sum()//124

    model_name = train_config['model_name']
    model= model_factory(model_name=model_name,config=train_config,sample_data=input_dict['train']) 
    weight_file = os.path.join(log_path,'model.h5')
    if model_name=='causal' or model_name=='causal_ablate': 
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='x_loss', min_delta=0.01, factor=0.01, patience=3,
        min_lr=1e-12,verbose=True)
        weight_file = os.path.join(log_path,'model.h5')
        callbacks = [reduce_lr]
        train_kwargs = {
        'steps_per_epoch':steps_per_epoch_train,
        'steps_per_epoch_val':steps_per_epoch_val,
        'verbose': True,
        'callbacks':callbacks,
        'vae_epoch':train_config['vae_epoch'],
        'x2u_epoch':train_config['x2u_epoch'],
        'xu2y_epoch':train_config['xu2y_epoch'],
        'calib_epoch':train_config['calib_epoch'],
        "vae_callbacks":None
        }
        model.fit(ds_dict_source['train'], ds_dict_source['val'], ds_dict_source['tune'], #The current run updates external not mayo
         **train_kwargs)
        #model.fit(ds_dict_source['train'], ds_dict_source['val'], ds_dict_source['external'], #The current run updates external not mayo
        # **train_kwargs)
        histories = pd.DataFrame(model.vae.history.history) 
        history_path = os.path.join(log_path,'causal_model_history.csv')
        histories.to_csv(history_path,index=False)
        if model_name=='causal':
            histories = pd.DataFrame(model.model_xu2y.history.history) 
        else: 
            histories = pd.DataFrame(model.model_risk.history.history) 
        history_path = os.path.join(log_path,'causal_model_xu2y_history.csv')
        histories.to_csv(history_path,index=False)
    if model_name=='baseline':
        weight_file = os.path.join(log_path,'model.h5')
        val_batch_size = 124
        val_freq = 1 
        val_data = ds_dict_source['val'].map(my_ret_func)
        val_size = colon_df[colon_df['split']=='internal-val'].copy().shape[0]
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', min_delta=0.01, factor=0.1, patience=10,
        min_lr=1e-12)
        callbacks = [reduce_lr] 
        if trial is None: 
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=weight_file,monitor='val_loss',save_best_only=True,save_weights_only=True,save_freq='epoch',verbose=1))
        train_kwargs = {
        'epochs':train_config['epochs'],
        'callbacks':callbacks,
        'steps_per_epoch':steps_per_epoch_train,
        'verbose':verbose,
        'validation_batch_size':val_batch_size,
        'validation_steps':val_size//val_batch_size,
        'validation_freq':val_freq,
        'validation_data':val_data
        } 
        history = model.fit(train_set,**train_kwargs)
        plt.figure(dpi=300)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['train Loss','Val Loss'])
        if trial is None: 
            plt.savefig(os.path.join(log_path,'train_graph.png'))
            model.load_weights(weight_file)
        val_loss = min(history.history['val_loss'])
    if trial:  
        return val_loss 
    if model_name=='causal' or model_name=='causal_ablate': 
        split_dfs = list() 
        val_steps=  steps_per_epoch_val
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
    else: 
        split_dfs = list() 
        base_hazard =  calculateBaselineHazard(input_dict['train'],event_col='event',time_col='time_to_event') 
        for split in test_ds_dict_source.keys(): 
            predictions = list() 
            for encode_in in iter(test_ds_dict_source[split].map(get_x_only)): #TODO make it so that  we use the baseline hazard and adjust accordingly at inference time 
                hazard = model.predict(encode_in)
                predictions.append(hazard)
            all_preds= np.vstack(predictions)
            risk = np.exp(all_preds)
            ht =  base_hazard*risk 
            all_preds = 1-np.exp(-1*ht.cumsum(axis=1))
            for k in range(all_preds.shape[1]): 
                input_dict[split][f'risk_{k}']= all_preds[:,k]
            split_df = pred_dict_to_frame(input_dict[split],train_config) 
            split_df['split'] = split
            split_dfs.append(split_df)
        test_df = pd.concat(split_dfs)
        csv_path = os.path.join(log_path,'model_infer.csv') 
        test_df.to_csv(csv_path,index=False) 
    if trial is None: 
        config_save_path = os.path.join(log_path,'orig_config.json') 
        with open(config_save_path,'w') as f : 
            json.dump(train_config,f)
    else: 
        return val_loss
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
        study = opt.create_study("sqlite:///my_param_search.db",study_name='search',direction='minimize',load_if_exists=True)
        wrap_main = lambda x: main(conf,x)
        study.optimize(wrap_main,n_trials=100) 
    else: 
        main(conf)
if __name__=='__main__':
    _parse()

