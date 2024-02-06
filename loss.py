import tensorflow as tf 
import pdb 
class SurvivalLoss(tf.keras.losses.Loss):
    def __init__(self) -> None:
        super(SurvivalLoss,self).__init__()
    def call(selfs,te,output_probs):
        time_to_event = te[:,0]
        event = te[:,1]
        return my_survival_loss(output_probs,time_to_event,event)
def my_survival_loss(output_probs,time_to_event,event): 
    time_to_event = tf.squeeze(time_to_event)
    output_probs = tf.squeeze(output_probs)
    event = tf.squeeze(event)
    idx = tf.argsort(time_to_event,direction='DESCENDING')
    time_to_event = tf.gather(time_to_event,idx)
    output_probs = tf.gather(output_probs,idx)
    event = tf.gather(event,idx)
    batch_size = event.shape[0]  
    gamma = tf.reduce_max(output_probs)
    observed_deaths = tf.math.reduce_sum(event) 
    y_h_hr = tf.exp(output_probs-gamma) 
    y_h_cumsum = tf.math.log(tf.cumsum(y_h_hr) ) +gamma 
    numerator= tf.reduce_sum(tf.math.multiply(tf.subtract(output_probs,y_h_cumsum),event)) 
    all_terms = numerator/observed_deaths 
    return -1* all_terms