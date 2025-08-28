# MAC441 Readme
Tutorial resources for MAC441 module "Resilient water infrastructure design"

Below is the procedure to install things from the command line. Refer to the document on Blackboard for "less geeky" install.

## How to clone and update the repository to your Uni drive U:\Users\<username>\
where <username> is your uni login (e.g., ciXXXX)

=> In the Search bar of your machine, type `cmd` to open a command prompt. This can be called "Command Prompt" or "Windows Powershell Command Prompt" or similar.

=> Navigate to `U:\Users\<username>` if needed, using the `cd Users` then `cd <username>` prompts.

=> If it is the first time your are importing this folder type
`git clone https://github.com/charlesrouge/mac441.git`

=> If it is not navigate to the mac441 folder using `cd mac441`then
`git pull origin main`

Note: there is a need to install git and that's not the case on all machines.


## How to run tutorials on Jupyter Notebook

1) Open Jupyter Notebook and identify the directory it starts in. For most machines this is
C:\Users\<username>

2) Still from the command line, exit the mac441 folder to be in C:\Users\<username>, with 
`cd ..`

3) Copy the mac441 folder to that directory using:
`Xcopy mac441 C:\Users\<username> /E`

Then press `D` if prompted to type `F` or `D`
Press `All` if prompted `Yes/No/All`
Note `/E` is needed to copy subfolders and files within them; otherwise only files in the main folder are copied.

4) The `mac441` folder should have magically appeared on the Jupyter Notebook main menu!


## How to install the virtual environment and run it on Jupyter Notebook

=> To create the virtual environment from the yml file: open the directory containing the `environment.yml` file and type in terminal
`conda env create -f environment.yml`

NOTE: Running this command may take long time i.e 10-15 mins on the University machine. Wait patiently. Once completed, you do not need to run it again!

=> Or if you cannot create the environment on the University machine, still create a new environment and work in it. This will keep track of libraries used and versions as you go!
`conda env create --name mac441`

=> type in terminal
`python -m ipykernel install --user --name=mac441`

=> in Jupyer Notebook click on `Python 3 (ipykernel)` in the top right, and replace with `mac441` as a kernel
