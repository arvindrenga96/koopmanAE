    
# Define the base directory where project results are stored.
baseDirectory="/panfs/jay/groups/32/kumarv/tayal/Downloads/Projects/koopmanAE/results"

# Define the job subdirectory within the base directory.
jobDirectory="${baseDirectory}/$1"

# Create the job subdirectory.
mkdir -p "${jobDirectory}"

# Define the subfolders for each specific algorithm result within the job directory.
daeSubfolder="${jobDirectory}/DAE"
koopmanAeSubfolder="${jobDirectory}/C-DAE"
koopmanInnSubfolder="${jobDirectory}/KIA"
rnnSubfolder="${jobDirectory}/RNN"

# Create an array of subfolders.
subfolders=("$daeSubfolder" "$koopmanAeSubfolder" "$koopmanInnSubfolder" "$rnnSubfolder")

# Iterate over each subfolder in the array. If it doesn't exist, create it.
for subfolder in "${subfolders[@]}"; do
    if [ ! -d "$subfolder" ]; then
        mkdir "$subfolder"
    fi
done


##Model Parameters
epochs="600"
seed="$2"
theta="0.4"
batch="64"


#DAE
command_1="python driver.py --alpha 6 --dataset sst --lr 1e-2 --epochs 900 --batch $batch --folder $daeSubfolder --lamb 1 --steps 14 --bottleneck 10 --lr_update 30 200 600 --lr_decay 0.2 --pred_steps 180 --backward 0 --wd 5e-4 --seed 1 --data_version sst_omri"

python generate_script.py "$daeSubfolder" "$command_1" "${jobDirectory}/dae_script"
sbatch "${jobDirectory}/dae_script.sh"


#KOOPMAN_AE

command_2="python driver.py --alpha 6 --dataset sst --lr 1e-2 --epochs 900 --batch $batch --folder $koopmanAeSubfolder --lamb 1 --steps 14 --bottleneck 10 --lr_update 30 200 600 --lr_decay 0.2 --pred_steps 180 --backward 1 --steps_back 6 --nu 1e-1 --eta 1e-2 --wd 5e-4 --seed 1 --data_version sst_omri"

python generate_script.py "$koopmanAeSubfolder" "$command_2" "${jobDirectory}/koopman_script"
sbatch "${jobDirectory}/koopman_script.sh"

#KOOPMAN_AE_INN

command_3="python driver.py --alpha 6 --dataset sst --lr 1e-2 --epochs 900 --batch $batch --folder $koopmanInnSubfolder --lamb 1 --steps 14 --bottleneck 10 --lr_update 30 200 600 --lr_decay 0.2 --pred_steps 180 --backward 1 --steps_back 14 --nu 1e-1 --eta 0 --wd 5e-4 --model koopmanAE_INN  --seed 1 --data_version sst_omri"

python generate_script.py "$koopmanInnSubfolder" "$command_3" "${jobDirectory}/koopman_inn_script"
sbatch "${jobDirectory}/koopman_inn_script.sh"

#RNN_new

command_4="python driver.py --alpha 6 --dataset sst  --lr 1e-2 --epochs 900 --batch $batch --folder $rnnSubfolder --lamb 1 --steps 6 --lr_update 30 200 600 --lr_decay 0.2 --pred_steps 180 --backward 0  --wd 5e-4 --model ConvLSTM  --seed 1 --data_version sst_omri"

python generate_script.py "$rnnSubfolder" "$command_4" "${jobDirectory}/rnn_script"
sbatch "${jobDirectory}/rnn_script.sh"


while true; do
 if [ -f "$daeSubfolder/000_pred.npy" ] && [ -f "$koopmanAeSubfolder/000_pred.npy" ] && [ -f "$koopmanInnSubfolder/000_pred.npy" ] && [ -f "$rnnSubfolder/000_pred.npy" ]; then
   echo "All files exist! Running the Python script..."
   python /home/kumarv/tayal/Downloads/Projects/koopmanAE/plot_pred_error.py "$daeSubfolder/000_pred.npy" "$koopmanAeSubfolder/000_pred.npy" "$koopmanInnSubfolder/000_pred.npy" "$rnnSubfolder/000_pred.npy" "$jobDirectory"
   break  # Exit the loop when the condition is met
 fi
 echo "Sleeping for 100 Seconds"
 sleep 100  # Sleep for 100 seconds
done



