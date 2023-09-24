# Text-Classification
Task 6 of the assignment given by Kaiburr

Task 6: Perform a Text Classification on the consumer complaint dataset into the following categories: 
0) Credit reporting, repair, or other
1) Debt collection
2) Consumer Loan
3) Mortgage

To run the code:

Option 1: Google Collab

Please copy the entire code from github and paste it into a Google Collab environment. The dataset "complaints.csv" should be located in '/content/drive/My Drive/complaints.csv'.

Runtime -> Change Runtime Type -> Hardware accelerator -> T4 GPU

Option 2: Jupyter Notebook

Hide the part of the code mentioning Google Drive and unhide the function which reads the file from the desktop

{ file_path = 'C:/Users/Aakash/Desktop/Kaiburr/Complaints/complaints.csv' original = pd.read_csv(file_path) }

You can change the file_path to your preferred location.
