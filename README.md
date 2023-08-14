
# Running the Project - Pre-processing Dataset and Running the Code

## Setting Up Environment
First, using Anaconda, set up a new environment called "topic_taxanomy" with the following commands. Note that a GPU will be required to properly run the code.

	conda create -n topic_taxonomy python==3.8
	conda activate topic_taxonomy
	conda install pandas
	conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
	conda install -c conda-forge bertopic

If the KCL HPC is being used, make sure you request a GPU as well as enough memory to run the code. A sample HPC GPU request can be seen below (this will open a new bash terminal with gpu access)

	srun -p gpu --gres gpu --mem=256G --ntasks 5 --pty /bin/bash
	 
## Pre-processing
From the dataset, that can be found [here](https://drive.google.com/drive/folders/1gWwm6k8K_1MmeaIVmywDcaPyR78Y-h7V), download "normalization_text" and extract the zip inside the "dataset" folder. Then, download the "report_summary_extracted_data.jsonl" and place that in the "dataset" folder as well. Next, run the following commands.

	cd ./src
	python preprocess.py
	python classify_esg.py

This will convert the report jsons into a list of documents, as well as classify all of them into either Environmental, Social, and Governance (first level of our taxonomy). This only needs to be run once as the results are saved into two files,  "./dataset/docs.pkl" and "./dataset/esg_classification.pkl," that will be used in the remained of our code as well.

## Running code 
Finally, we can run the final part of the code now. Run the following command to run the code that will generate the second-level and third-level topics.

	python run_topic_taxonomy.py

This will generate the final taxonomy, generate a visualization, as well as assign the documents to all the relevant topics.
Three files are generated:
	1.  An interactive visualization of the whole taxonomy (all three levels) is generated and saved into a file called "complete_taxonomy_visualization.html"
	2. The taxonomy converted into a JSON, as well as the id of docs that have been assigned to that JSON, saved into a file called "topic_taxonomy.json"
	3. A list of JSON that contain the documents, their report url, and their page number. The index matches the doc id from the previous file, "topic_taxonomy.json," and so the document and its info can be accessed from here. The file is called "topic_taxonomy_docs.json"
