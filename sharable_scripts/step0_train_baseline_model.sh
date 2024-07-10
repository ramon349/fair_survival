
root_dir="/your_path_to_this_folder/"
cd "${root_dir}/fair_survival/ColonCancer_survival"
model_path="${root_dir}baseline_model_outputs/model_dir/"
train_data="${root_dir}/datasets/cox_model_tr.csv"
val_data="${root_dir}/datasets/cox_model_val.csv"
test_file="${root_dir}/datasets/causal_encoder_encoded_dataset.csv"
python3 main.py  --modelpath "${model_path}" --traindata "${train_data}" \
--validationdata "${val_data}" \
--testdata "${test_file}" \
--savepath "${model_path}" \
--flag Train

test_file_save="${root_dir}/baseline_model_outputs/cox_model_preds.csv"
python3 main.py  --modelpath "${model_path}" --testdata  "${test_file}" --flag Test --savepath "${test_file_save}" 
