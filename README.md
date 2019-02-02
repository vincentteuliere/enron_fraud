In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. 

In this project, using machine learning tehcniques, we'll attemp to predict and spot persons of interest(POI) based on financial and email data made public as a result of the Enron scandal. A person of interest in the fraud case, means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. 

Ressources available for this classifcation are:
- POI labels: the list of persons of interest,which has been hand-generated based on various public ressources. 
- financial features extracted from a PDF file summarizing all payments and stocks values of each employee involved in the lawsuit.
- emails features describing the volume of emails exchanged with POI , which have been processed from email database

All these data have already been pre-processed into a Python dictionnary located in final-project/final_project_dataset.pkl

- Refer to final_project/explore.ipynb fore details on the exploration & final report.
- Final classifier can be generated & cross-validated by executing final_project/poi_id.py
