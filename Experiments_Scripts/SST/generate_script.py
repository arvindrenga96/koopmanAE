import sys

def generate_script(folder_path, command, script_name):
    content = '''#!/bin/bash -l

###############################

# SETUP RESOURCE
#SBATCH -A jusun
#SBATCH --time=4:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --partition=apollo_agate
#SBATCH --mem=10gb
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=tayal@umn.edu
#SBATCH --output={folder_path}/1.out
#SBATCH --error={folder_path}/1.err

###############################

# FIXED COMMANDS

cd /home/kumarv/tayal/Downloads/Projects/koopmanAE/

export PATH=/home/kumarv/tayal/anaconda5/envs/main_a100/bin:$PATH

{command}
'''.format(folder_path=folder_path, command=command)

    with open('{}.sh'.format(script_name), 'w') as file:
        file.write(content)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Please provide the folder path, command, and script name as command-line arguments.')
    else:
        folder_path = sys.argv[1]
        command = sys.argv[2]
        script_name = sys.argv[3]
        generate_script(folder_path, command, script_name)