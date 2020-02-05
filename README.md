# question_answering
Finetuning ALBERT on SQUAD dataset
## User guide
1) Copy squad *.json files in the data/squad folder
2) Set up environment with tensorflow==2.1 transformers 
(docker also usable with ./build.sh and ./start_container.sh, but change mounting directory in ./start_container.sh!)
3) Save transformers ALBERT model in local directory (especially if using docker!)
4) Change paths in train.py script
