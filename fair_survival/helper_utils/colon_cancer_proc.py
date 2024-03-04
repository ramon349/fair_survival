
from sklearn.preprocessing import OneHotEncoder 
import numpy as np 

class ColonDfProcessor():
    def __init__(self,encoders,merge_stages=False) -> None:
        self.encoders = encoders
        self.merge_stages = merge_stages 
    def fit(self,df):
        for e in self.encoders.keys(): 
            self.encoders[e].fit(df[[e]]) #do this so the 
            #encoder sees a dataframe of shape Nx1 and preserves the name 
    def transform(self,df): 
        for e in self.encoders.keys(): 
            feat_names =  self.encoders[e].get_feature_names_out()
            out_arr = self.encoders[e].transform(df[[e]])
            for i,e in enumerate(feat_names):
                df[feat_names[i]] = out_arr[:,i] #copy over the one hot array 
            

#TODO: would be nice to define procesisng functions as objects that can handle data transforms
def processor(my_df,merge_stages=False,encoders=None):  
    """ processing funciton to encode data as either one hot or multimodal 
    merge_stages -->  we can treat stages  1 and 2 as being 1 stage
    encoders --> calling with encoders none  will  create an encoder object and return it 
    if a dictionary of encoders is provided we will simply transform the data with no fitting 
    """
    if encoders:
        stage_enc = encoders['stage']
        yval_enc = encoders['yval']
        mmr_encoder = encoders['mmr'] 
    else:
        stage_enc = OneHotEncoder()
        yval_enc = OneHotEncoder() 
        mmr_encoder = OneHotEncoder()
    if merge_stages: 
        my_df.loc[my_df['Stage']==1,'Stage'] = 1
        my_df.loc[my_df['Stage']==2,'Stage'] = 1
        my_df.loc[my_df['Stage']==3,'Stage'] = 2
    if not encoders:
        stage_enc.fit(my_df['Stage'].values.reshape(-1,1))
    stage_encodings = stage_enc.transform(my_df['Stage'].values.reshape(-1,1)).todense()
    for i,e in enumerate(my_df['Stage'].unique()): 
        my_df[f'Stage_{e}'] = stage_encodings[:,i] 
    center_codes = sorted(my_df['center'].unique())
    for i,e in enumerate(center_codes): 
        my_df[f'center_{e}'] = (my_df['center']==e).astype(int)
    my_df['center_ext'] = np.logical_not(my_df['center'].isin(center_codes)).astype(int)
    #do the mmr status encoding 
    mmr_col = 'MMR status, 0=MMRP, 0=MMRD'
    if not encoders: 
        mmr_encoder.fit(my_df[mmr_col].values.reshape(-1,1))
    mmr_encodings = mmr_encoder.transform(my_df[mmr_col].values.reshape(-1,1)).todense()
    for i,e in enumerate(my_df[mmr_col].unique()): 
        my_df[f'mmr_{e}'] = mmr_encodings[:,i] 
    if not encoders: 
        encoders = {} 
        encoders['stage'] = stage_enc
        encoders['yval'] = yval_enc 
        encoders['mmr']=mmr_encoder
    return my_df,encoders 
