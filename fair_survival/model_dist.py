
import tensorflow as tf 
import fair_survival.loss as loss_funcs
from fair_survival.helper_utils.data_proc import get_dims,get_dims_mammo
import ml_collections as mlc 
from  .helper_utils.data_proc import calculateBaselineHazard
from .latent_shift_adaptation.methods.vae import gumbelmax_graph
DEFAULT_LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True) 
import pdb 

def model_factory(model_name,config=None,sample_data=None): 
    if model_name=='baseline': 
        return get_baseline_model() 
    if model_name=='causal': 
        return build_latent_shift_model(sample_data=sample_data,config=config)
    raise ValueError("Incorrect Model name")
def get_baseline_model(load_path=None):
    #defining model graph 
    model_input = tf.keras.Input(shape=(15,))
    x = model_input
    #Block 1
    x= tf.keras.layers.Dense(6,activation=None)(x)
    x = tf.keras.layers.ReLU()(x)
    x= tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.1)(x)
    x=tf.keras.layers.Dropout(rate=0.024)(x)
    #block2
    x = tf.keras.layers.Dense(3,activation=None)(x)
    x = tf.keras.layers.ReLU()(x)
    x= tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.1)(x)
    x= tf.keras.layers.Dropout(rate=0.024)(x)
    #prediciton head 
    model_output = tf.keras.layers.Dense(1,activation=None)(x)
    #optimizers  and losses 
    opt = tf.keras.optimizers.Adam(learning_rate=0.01) 
    model = tf.keras.models.Model(model_input,model_output)
    model.compile(optimizer=opt,loss=loss_funcs.SurvivalLoss())
    if load_path: 
        model.load_weights(load_path)
    return model 

def mlp(num_classes, width, input_shape, learning_rate,
        loss=DEFAULT_LOSS, metrics=[]): #i changed the default to be None so it would yell at me 
  """Multilabel Classification."""
  model_input = tf.keras.Input(shape=input_shape)
  # hidden layer
  if width:
    x = tf.keras.layers.Dense(
        width, use_bias=True, activation='relu'
    )(model_input)
  else:
    x = model_input
  model_outuput = tf.keras.layers.Dense(num_classes,
                                        use_bias=True,
                                        activation="linear")(x)  # get logits
  opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
  model = tf.keras.models.Model(model_input, model_outuput)
  model.build(input_shape)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)
  return model

def build_x2u(num_classes,width,input_shape,learning_rate,loss=DEFAULT_LOSS,metrics=[]): 
  model_input = tf.keras.Input(shape=input_shape)
  x = tf.keras.layers.Dense(width,use_bias=True,activation='relu')(model_input)
  x = tf.keras.layers.Dense(12,use_bias=True,activation='relu')(x)
  model_output = tf.keras.layers.Dense(num_classes,use_bias=True,activation='linear')(x) 
  opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
  model = tf.keras.models.Model(model_input, model_output)
  model.build(input_shape)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)
  return model 


def mlp_time(input_shape):
    model = get_model(input_shape)  
    return model 
def get_model(input_shape):
    #defining model graph 
    model_input = tf.keras.Input(shape=(input_shape,))
    x = model_input
    #Block 1
    x= tf.keras.layers.Dense(6,activation=None)(x)
    x = tf.keras.layers.ReLU()(x)
    x= tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.1)(x)
    x=tf.keras.layers.Dropout(rate=0.024)(x)
    #block2
    x = tf.keras.layers.Dense(3,activation=None)(x)
    x = tf.keras.layers.ReLU()(x)
    x= tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.1)(x)
    x= tf.keras.layers.Dropout(rate=0.024)(x)
    #prediciton head 
    model_output = tf.keras.layers.Dense(1,activation=tf.keras.activations.sigmoid)(x)
    #optimizers  and losses 
    opt = tf.keras.optimizers.Adam(learning_rate=0.01) 
    model = tf.keras.models.Model(model_input,model_output)
    model.compile(optimizer=opt,loss=loss_funcs.SurvivalLoss())
    return model 
def get_model_time_mammo(input_dim, load_path=None):
    #defining model graph 
    model_input = tf.keras.Input(shape=(input_dim,))
    x = model_input
    #Block 1
    x= tf.keras.layers.Dense(256,activation=None)(x)
    x = tf.keras.layers.ReLU()(x)
    x= tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.1)(x)
    x=tf.keras.layers.Dropout(rate=0.024)(x)
    #block2
    x = tf.keras.layers.Dense(128,activation=None)(x)
    x = tf.keras.layers.ReLU()(x)
    x= tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.1)(x)
    x= tf.keras.layers.Dropout(rate=0.024)(x)
    #block3
    x = tf.keras.layers.Dense(64,activation=None)(x)
    x = tf.keras.layers.ReLU()(x)
    x= tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.1)(x)
    x= tf.keras.layers.Dropout(rate=0.024)(x)
    #block 4
    x = tf.keras.layers.Dense(32,activation=None)(x)
    x = tf.keras.layers.ReLU()(x)
    x= tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=0.1)(x)
    x= tf.keras.layers.Dropout(rate=0.024)(x)
    #prediciton head 
    model_output = tf.keras.layers.Dense(1,activation=None)(x)
    #optimizers  and losses 
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001) 
    model = tf.keras.models.Model(model_input,model_output)
    model.compile(optimizer=opt,loss=loss_funcs.SurvivalLoss())
    if load_path: 
        model.load_weights(load_path)
    return model 

def figure_dims(sample_data,config):
    if config['dataset']=='mammo':
        return get_dims_mammo(sample_data) 
    if config['dataset']=='colon':
        return get_dims(sample_data)
def build_latent_shift_model(sample_data,config,train_d=None):
    #num classes whould always be 1
    latent_dim =  config['latent_dim'] # 21 
    width = config['width'] #  8 
    kl_coeff = config['kl_coeff']# 3 
    learning_rate = config['learning_rate'] # 1e-3
    x_dim,c_dim,w_dim = figure_dims(sample_data=sample_data,config=config)
    num_classes = 1 #TODO make this not be hardcoded.  Right now i want to make it adaptable
    encoder = mlp(num_classes=latent_dim, width=width,
            input_shape=(x_dim + c_dim + w_dim + num_classes+1+1),
            learning_rate=learning_rate,
            metrics=['accuracy']) #TODO: Why is accruacy a metric here

    model_x2u = mlp(num_classes=latent_dim, width=width, input_shape=(x_dim,),
                    learning_rate=0.01,
                    metrics=['accuracy'])
    if config['dataset']=='colon':
        model_xu2y = mlp_time(input_shape=(x_dim+latent_dim))
    if config['dataset']=='mammo': 
        model_xu2y = get_model_time_mammo(input_dim=(x_dim+latent_dim))
    vae_opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    dims = mlc.ConfigDict()
    dims.x = x_dim
    dims.y = 1
    dims.c = c_dim
    dims.w = w_dim
    dims.u =  latent_dim
    dims.t = 1
    dims.e = 1
    print(dims)
    if config['dataset']=='mammo': 
        base_haz = calculateBaselineHazard(train_d)
    else: 
        base_haz = calculateBaselineHazard(sample_data,event_col='event',time_col='time_to_event')
    evaluate = tf.keras.metrics.BinaryCrossentropy()
    pos = mlc.ConfigDict()
    pos.x, pos.y, pos.c, pos.w,pos.t,pos.e= 0, 1, 2, 3, 4, 5 #t # TODO: you can do the swap of tiem to event here by simply swapping out  some vlaues 
    clf = gumbelmax_graph.Method(encoder, width, vae_opt,
                    model_x2u, model_xu2y, 
                    dims, latent_dim, None,
                    kl_loss_coef=kl_coeff,
                    num_classes=num_classes, evaluate=evaluate,
                    dtype=tf.float32, pos=pos,baselineHazards=base_haz,c_weights=config['c_weights'],w_weights=config['w_weights'])
    return clf 
