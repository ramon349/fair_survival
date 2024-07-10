# Install info 
Required python is 3.11 

requirements.txt is still under work 

# Reproducing this work. (Data setup) 
1. Run notebooks labeled step 0a and 0b in the  notebooks directory 
2. You will find that they  produce a csv file called causal_encoder_encoded_dataset.csv this is the one we will use for all our development 
3. You will now run the baseline cox proportional hazards models trained by the previous group 
    ```bash 
    bash ./sharable_scripts/step0_train_baseline_models.sh  
    ```
4. Training our model 
    ```bash 
    bash ./sharable_scripts/step0_train_baseline_models.sh  
    ```

# Side notes: 
    - it might be unfair to give the mayo data as an additional tuning set. We might want to train the baseline model with the inclusion of the tuning set
    - this should also count for the baseline pycoxmdoel 

