# Automated Extraction of Medical Risk Factors For Life Insurance Underwriting

Life insurance underwriting considers an applicant’s medical risk factors, usually provided inside free-text documents. New insurance-specific Natural Language Processing (NLP) models can automatically extract material medical history and risk factors from such documents. This joint Solution Accelerator with John Snow Labs makes it easy to implement this in practice – enabling a faster, more consistent, and more scalable underwriting experience. This tutorial will cover:
- The end-to-end solution architecture on Databricks, from data ingestion to dashboarding
- Easily analyze free-text documents to extract medical history & risk factors using NLP
- Executable Python notebooks implementing the solution that you can start from today

We will get the following list of medical risk factors from unstructured clinical notes using Spark NLP models and tools and make analysis on the results.<br>


- Basic Profile
    - ✅ Age
    - ✅ Gender
    - ✅ Weight
    - ✅ Height
    - ✅ Race/Ethnicity
    - ✅ Disability
- Personal History
    - ✅ Medical records (ICD-10-CM)
    - ✅ Prescription history (RxNorm)
    - ✅ Actions of prescriptions (Action Mapper)
    - ✅ Family health history (ICD-10-CM + Assertion)
- Lifestyle
    - ✅ Profession
    - ✅ Marital status
    - ✅ Housing
    - ✅ Smoking
    - ✅ Alcohol
    - ✅ Substance
- Diseases
    - ✅ Asthma and breathing problems
    - ✅ Heart disease, including heart attacks and angina
    - ✅ High cholesterol
    - ✅ High blood pressure
    - ✅ Hypertension
    - ✅ Cancer
    - ✅ Strokes, including mini-strokes and brain haemorrage
    - ✅ Anxiety
    - ✅ Depression
    - ✅ Diabetes
    - ✅ Obesity
    - ✅ Epilepsy
    - ✅ Cerebral palsy and other neurological conditions
    - ✅ Kidney diseases


## License
Copyright / License info of the notebook. Copyright [2023] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.

|Library Name|Library License|Library License URL|Library Source URL|
| :-: | :-:| :-: | :-:|
|Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
|Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
|Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
|MatPlotLib | | https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE | https://github.com/matplotlib/matplotlib|
|Seaborn |BSD 3-Clause License | https://github.com/seaborn/seaborn/blob/master/LICENSE | https://github.com/seaborn/seaborn/|
|Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
|Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
|Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|


|Author|
|-|
|Databricks Inc.|
|John Snow Labs Inc.|


## Disclaimers
Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.


## Instruction
To run this accelerator, set up JSL Partner Connect [AWS](https://docs.databricks.com/integrations/ml/john-snow-labs.html#connect-to-john-snow-labs-using-partner-connect), [Azure](https://learn.microsoft.com/en-us/azure/databricks/integrations/ml/john-snow-labs#--connect-to-john-snow-labs-using-partner-connect) and navigate to **My Subscriptions** tab. Make sure you have a valid subscription for the workspace you clone this repo into, then **install on cluster** as shown in the screenshot below, with the default options. You will receive an email from JSL when the installation completes.

<br>
<img src="https://raw.githubusercontent.com/databricks-industry-solutions/oncology/main/images/JSL_partner_connect_install.png" width=65%>

Once the JSL installation completes successfully, clone this repo into a Databricks workspace. Attach the `RUNME` notebook to any cluster and execute the notebook via `Run-All`. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.
