
# Define the base directory where project results are stored.
baseDirectory="/panfs/jay/groups/32/kumarv/tayal/Downloads/Projects/koopmanAE/results"

# Define the job subdirectory within the base directory.
jobDirectory="${baseDirectory}/$1"

# Create the job subdirectory.
mkdir -p "${jobDirectory}"

# Define the subfolders for each specific algorithm result within the job directory.
daeSubfolder="${jobDirectory}/DAE"
koopmanAeSubfolder="${jobDirectory}/Koopman_AE"
koopmanInnSubfolder="${jobDirectory}/KOOPMAN_AE_INN"
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
noise="0.03"


#DAE
command_1="python driver.py --alpha 1 --dataset lorenz --noise $noise --lr 1e-2 --epochs $epochs --batch 64 --folder $daeSubfolder --lamb 1 --steps 8 --bottleneck 6  --lr_update  30 200 400 500 --lr_decay 0.2 --pred_steps 1000 --backward 0 --seed $seed"

python generate_script.py "$daeSubfolder" "$command_1" "${jobDirectory}/dae_script"
sbatch "${jobDirectory}/dae_script.sh"


#KOOPMAN_AE

command_2="python driver.py --alpha 1 --dataset lorenz --noise $noise --lr 1e-2 --epochs $epochs --batch 64 --folder $koopmanAeSubfolder --lamb 1 --steps 8 --bottleneck 6  --lr_update  30 200 400 500 --lr_decay 0.2 --pred_steps 1000 --backward 1 --steps_back 8 --nu 1e-1 --eta 1e-2 --seed $seed"

python generate_script.py "$koopmanAeSubfolder" "$command_2" "${jobDirectory}/koopman_script"
sbatch "${jobDirectory}/koopman_script.sh"

#KOOPMAN_AE_INN
command_3="python driver.py --alpha 1 --dataset lorenz --noise $noise --lr 1e-2 --epochs $epochs --batch 64 --folder $koopmanInnSubfolder --lamb 1 --steps 8 --bottleneck 6  --lr_update  30 200 400 500 --lr_decay 0.2 --pred_steps 1000 --backward 1 --steps_back 8 --nu 1 --eta 0 --seed $seed --model koopmanAE_INN"

python generate_script.py "$koopmanInnSubfolder" "$command_3" "${jobDirectory}/koopman_inn_script"
sbatch "${jobDirectory}/koopman_inn_script.sh"

#RNN_new

command_4="python driver.py --alpha 1 --dataset lorenz --noise $noise --lr 1e-2 --epochs $epochs --batch 64 --folder $rnnSubfolder --lamb 1 --steps 8 --bottleneck 6  --lr_update  30 200 400 500 --lr_decay 0.2 --pred_steps 1000 --backward 0 --seed $seed --model LSTM"

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



