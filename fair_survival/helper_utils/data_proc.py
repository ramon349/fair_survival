import tensorflow as tf  
from lifelines import KaplanMeierFitter
import pdb 
import numpy as np 

def calculateBaselineHazard(train_df,event_col='surv_1',time_col='surv_0') :
    km = KaplanMeierFitter()
    km.fit(train_df[time_col],train_df[event_col]) #TODO: change the colon script to pass in 
    #event_col = event,  time_col = time_to_event
    surv_base = km.survival_function_['KM_estimate'].values
    haz_base = 1-surv_base
    return haz_base
def convert_3months( T):
    return int(T/2)*2
def encode( train):
    ## Time to event till 4 years - 48 months
    T = []
    E = []

    for i in range(train.shape[0]):
        if train.iloc[i]['Time to recurrence or last follow-up (months)'] < 60: # 5 years
            T.append(convert_3months(train.iloc[i]['Time to recurrence or last follow-up (months)']))
            E.append(train.iloc[i]['Recurrence status at last F/up, 0=No recurrence, 1=recurrence'])
        else:
            T.append(60)
            E.append(0)
    train['Event'] = E
    train['Time to event'] = T
    return train
def get_input(*batch,inputs=None): 
    stack = tf.cast(batch[inputs[0]],tf.float64)
    for c in inputs[1:]:
        stack = tf.concat(
            [stack,tf.cast(batch[c],tf.float64)],axis=1
        )
    return stack
def my_ret_func(*batch,x_location=[0],y_location=[1,2]):
    x = get_input(*batch,inputs=x_location)
    te = get_input(*batch,inputs=y_location)
    return x,te 
def get_x_only(*batch): 
    x = get_input(*batch,inputs=[0])
    return x 

def get_dims(tr_dict): 
    #input sould be the tf batch dataset object. 
    # iteration is handeled inside my own code 
    x_dim =  tr_dict['x'].shape[1]
    c_dim =  tr_dict['c'].shape[1]
    w_dim = tr_dict['w_one_hot'].shape[1]
    return x_dim,c_dim,w_dim
def make_set_by_split(my_df,split,feat_names,config=None):
    df = my_df[my_df['split']==split].copy()
    print(f"Split {split} has shape {df.shape[0]}")
    my_feat_dict= {} 
    my_feat_dict['x'] = df[feat_names].values 
    event = df['Event'].values.reshape(-1,1)
    print(df['Event'].value_counts())
    not_event =  (event==0).astype(int)
    encoded = np.hstack((not_event,event)).astype(float)
    my_feat_dict['y_one_hot'] =  df['Event'].values.reshape(-1,1) #encoded[:,1].reshape(-1,1) #TODO check if you can revert to just having one column 
    my_feat_dict['w_one_hot'] = df[[e for e in df.keys() if e.startswith('center_')]] .values.astype(float)
    my_feat_dict['c'] = df[[e for e in df.keys() if e.startswith('Stage_')]] .values.astype(float)
    my_feat_dict['time_to_event'] = df['Time to event'].values.reshape(-1,1).astype(float)
    my_feat_dict['event'] = df['Event'].values.reshape(-1,1).astype(float)
    my_feat_dict['Study ID'] = df['Study ID'].values.reshape(-1,1)
    return my_feat_dict
def get_dims_mammo(tr_dict): 
    #input sould be the tf batch dataset object. 
    # iteration is handeled inside my own code  
    sample = next(iter(tr_dict))
    x_dim = sample[0].shape[1]
    c_dim =   sample[2].shape[1]
    w_dim =  sample[3].shape[1]
    return x_dim,c_dim,w_dim