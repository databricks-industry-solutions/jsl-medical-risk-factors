# Databricks notebook source
# MAGIC %md
# MAGIC # Automated extraction of medical risk factors for life insurance underwriting
# MAGIC  
# MAGIC Life insurance underwriting considers an applicant’s medical risk factors, usually provided inside free-text documents. New insurance-specific Natural Language Processing (NLP) models can automatically extract material medical history and risk factors from such documents. This joint Solution Accelerator with John Snow Labs makes it easy to implement this in practice – enabling a faster, more consistent, and more scalable underwriting experience. This webinar will cover:
# MAGIC - The end-to-end solution architecture on Databricks, from data ingestion to dashboarding
# MAGIC - Easily analyze free-text documents to extract medical history & risk factors using NLP
# MAGIC - Executable Python notebooks implementing the solution that you can start from today

# COMMAND ----------

# MAGIC %md
# MAGIC ## Insurance Risk Factors

# COMMAND ----------

# MAGIC %md
# MAGIC - Basic Profile
# MAGIC     - ✅ Age 
# MAGIC     - ✅ Gender 
# MAGIC     - ✅ Weight 
# MAGIC     - ✅ Height 
# MAGIC - Personal History
# MAGIC     - ✅ Medical records (ICD)
# MAGIC     - ✅ Prescription history (RxNorm) 
# MAGIC     - ✅ Actions of prescriptions (Action Mapper)
# MAGIC     - ✅ Family health history (ICD + Assertion)
# MAGIC     - Criminal history (Excluded - received from authorities)
# MAGIC     - Driving history (Excluded - received from authorities)
# MAGIC - Lifestyle
# MAGIC     - ✅ Profession
# MAGIC     - ✅ Marital status
# MAGIC     - ✅ Smoking 
# MAGIC     - ✅ Alcohol 
# MAGIC     - ✅ Substance 
# MAGIC - Diseases
# MAGIC     - Asthma and breathing problems
# MAGIC     - Heart disease, including heart attacks and angina
# MAGIC     - High cholesterol
# MAGIC     - High blood pressure
# MAGIC     - Cancer
# MAGIC     - Strokes, including mini-strokes and brain haemorrage
# MAGIC     - Anxiety
# MAGIC     - Depression
# MAGIC     - Diabetes
# MAGIC     - Obesity
# MAGIC     - Epilepsy
# MAGIC     - Cerebral palsy and other neurological conditions
# MAGIC     - Kidney diseases

# COMMAND ----------

# MAGIC %md
# MAGIC **Initial Configurations**

# COMMAND ----------

import json
import os
import string
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.util import *
from sparknlp.annotator import *
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from sparknlp.pretrained import ResourceDownloader

pd.set_option("display.max_colwidth",0)
pd.set_option("display.max_columns",0)
pd.set_option('display.expand_frame_repr', False)

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions
# MAGIC 
# MAGIC We will define some helper functions to use in downstreaming tasks. 

# COMMAND ----------

def get_resolver_code(light_pipeline, text, output_col):
    # returns resolutions and codes
    res = light_pipeline.fullAnnotate(text)[0][output_col][0]
    res_code = res.result
    resolution = res.metadata['resolved_text']
    return res_code, resolution

# COMMAND ----------

def get_treatment_action(drug_rxnorm):
    # returns RxNorm codes and drug actions
    try:
        action = mapper_lp.fullAnnotate(drug_rxnorm[0])[0]['action'][0].result
    except: 
        action = "NONE"
    return (drug_rxnorm[1], action)

# COMMAND ----------

def get_occurence(df, label, pair_count):
    # returns the counts of the term occurence 
    d  = Counter()
    a = df[label].to_list()
    for sub in a:
        if len(a) < pair_count:
            continue
        sub.sort()
        for comb in combinations(sub, pair_count):
            d[comb] += 1
    return d

# COMMAND ----------

def get_normalized_name(df, column):
    # returns the normalized terms only
    normalized_names = []
    for i in df[column]:
        if len(i)>0:
            tmp_list = [j[1] for j in i]
        else:
            tmp_list = i
        normalized_names.append(tmp_list)
    return normalized_names

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download Clinical Notes
# MAGIC 
# MAGIC First we will create our folders and then download sample clinical notes.
# MAGIC 
# MAGIC In this notebook we will use the slightly modified version of transcribed medical reports in [www.mtsamples.com](www.mtsamples.com). 
# MAGIC 
# MAGIC You can download those reports by the script [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/healthcare_case_studies/mt_scrapper.py).

# COMMAND ----------

# create folders
dbutils.fs.mkdirs("dbfs:/databricks/driver/Insurence_Risk_Factors")
dbutils.fs.mkdirs("dbfs:/databricks/driver/Insurence_Risk_Factors/data")
dbutils.fs.mkdirs("dbfs:/databricks/driver/Insurence_Risk_Factors/results")

# COMMAND ----------

#download dataset
! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/healthcare-nlp/data/mt_data.csv -P /dbfs/databricks/driver/Insurence_Risk_Factors/data

# COMMAND ----------

# read data as a pandas dataframe
data = pd.read_csv('/dbfs/databricks/driver/Insurence_Risk_Factors/data/mt_data.csv')
data.head()

# COMMAND ----------

data.shape

# COMMAND ----------

# list of medical specialities in data
[i for i in data.medical_speciality.unique()]

# COMMAND ----------

# MAGIC %md
# MAGIC **We will choose some of the most related medical specialities that can be used in our insurance risk factor task.**

# COMMAND ----------

risky_diseases = ['Bariatrics', 'Cardiovascular_Pulmonary', 'Hematology_Oncology', 'General_Medicine', 'Neurology', 'Nephrology', 'Obstetrics_Gynecology', 'Psychiatry_Psychology', 'Radiology', 'Endocrinology']
data = data[data.medical_speciality.isin(risky_diseases)].reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC **Now we will get one clinical note sample from each medical speciality and work on this data.**

# COMMAND ----------

df_samples = pd.DataFrame()
for a, group in data.groupby('medical_speciality'):
    df_samples = pd.concat([df_samples, group.head(1)], ignore_index=True)

df_samples

# COMMAND ----------

# create spark dataframe 
df = spark.createDataFrame(df_samples)
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Initial NLP Pipeline Stages

# COMMAND ----------

documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
  .setInputCols(["document"]) \
  .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols(["sentence"])\
  .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extracting Diseases
# MAGIC 
# MAGIC We will download the `sparknlp_jsl` NER models and whitelist the labels that can be used as insurance risk factor.
# MAGIC 
# MAGIC Lets check the NER labels in `ner_jsl` model and whitelist the ones that are most related to our case. You can add other labels or remove some of them by editting the whitelist.

# COMMAND ----------

jsl_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_jsl")\
    .setLabelCasing('upper')

# COMMAND ----------

# ner_jsl labels

set([i.split("-")[1] for i in jsl_ner.getClasses() if i != "O"])

# COMMAND ----------

ner_jsl_entities = [
    'Age', 'Alcohol', 'Cerebrovascular_Disease', 'Diabetes',
    'Disease_Syndrome_Disorder', 'Drug_BrandName', 'Drug_Ingredient',
    'Employment', 'Heart_Disease', 'Hyperlipidemia','Hypertension',
    'Kidney_Disease', 'Obesity', 'Oncological', 'Procedure',
    'Smoking', 'VS_Finding', 'Drug', 'EKG_Findings', 'Height', 'ImagingFindings',
    'Overweight', 'Psychological_Condition', 'Substance', 'BMI', 
    'Total_Cholesterol', 'Weight', 'ImagingFindings', 'HDL', 'LDL', 'Race_Ethnicity' 
]
ner_jsl_entities = [a.upper() for a in ner_jsl_entities]

ner_posology_entities = ['DRUG']

ner_risks_entities = ['CAD','DIABETES','HYPERLIPIDEMIA','HYPERTENSION','MEDICATION','OBESE','SMOKER']

ner_sdoh_entities = [
    'Housing', 'Age', 'Alcohol','Employment','Mental_Health', 'Marital_Status',
    'Other_Disease','Smoking','Substance_Use', 'Disability', 'Race_Ethnicity'
]
ner_sdoh_entities = [a.upper() for a in ner_sdoh_entities]

ner_deid_entities = ["PROFESSION", "AGE"]

# COMMAND ----------

# risk factors
risk_factors_ner = MedicalNerModel.pretrained("ner_risk_factors", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_risks")\
    .setLabelCasing('upper')

risk_factors_ner_converter = NerConverterInternal()\
    .setInputCols(["sentence","token","ner_risks"])\
    .setOutputCol("ner_risks_chunk")\
    .setWhiteList(ner_risks_entities)

# general clinical terminology
jsl_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_jsl")\
    .setLabelCasing('upper')

jsl_ner_converter = NerConverterInternal()\
    .setInputCols(["sentence","token","ner_jsl"])\
    .setOutputCol("ner_jsl_chunk")\
    .setWhiteList(ner_jsl_entities)

# social determinants of health (sdoh)
sdoh_ner = MedicalNerModel.pretrained("ner_sdoh_slim_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_sdoh")\
    .setLabelCasing('upper')

sdoh_ner_converter = NerConverterInternal()\
    .setInputCols(["sentence","token","ner_sdoh"])\
    .setOutputCol("ner_sdoh_chunk")\
    .setWhiteList(ner_sdoh_entities)

# posology
posology_ner = MedicalNerModel.pretrained("ner_posology_large","en","clinical/models")\
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner_posology")\
    .setLabelCasing('upper')

posology_ner_converter = NerConverterInternal()\
    .setInputCols(["sentence","token","ner_posology"])\
    .setOutputCol("ner_posology_chunk")\
    .setWhiteList(ner_posology_entities)

# deidentification - Profession and Age labels only
deid_ner = MedicalNerModel.pretrained("ner_deid_generic_augmented","en","clinical/models")\
    .setInputCols(["sentence","token","embeddings"])\
    .setOutputCol("ner_deid")\
    .setLabelCasing('upper')

deid_ner_converter = NerConverterInternal()\
    .setInputCols(["sentence","token","ner_deid"])\
    .setOutputCol("ner_deid_chunk")\
    .setWhiteList(ner_deid_entities)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also add rule based approaches to our pipeline. Here we will use our context aware rule based component `ContextualParserApproach()` to support NER models for detecting `AGE` entities.

# COMMAND ----------

age = {
  "entity": "AGE",
  "ruleScope": "sentence",
  "matchScope":"token",
  "regex":"\\d{1,3}",
  "prefix":["age of", "age"],
  "suffix": ["-years-old", "years-old", "-year-old",
             "-months-old", "-month-old", "-months-old",
             "-day-old", "-days-old", "month old",
             "days old", "year old", "years old", 
             "years", "year", "months", "old"],
  "contextLength": 25,
  "contextException": ["ago", "last", "before", "spent", "later", "after"],
  "exceptionDistance": 12
}

with open('age.json', 'w') as f:
    json.dump(age, f)


age_contextual_parser = ContextualParserApproach() \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("age_cp") \
        .setJsonPath("age.json") \
        .setCaseSensitive(False) \
        .setPrefixAndSuffixMatch(False)\
        .setShortestContextMatch(True)\
        .setOptionalContextRules(False) 

age_chunk_converter = ChunkConverter() \
    .setInputCols(["age_cp"]) \
    .setOutputCol("chunk_age")

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will normalize the NER labels by creating a mapping dictionary and using this dictionary in `ChunkMergerApproach()`. In this way, we will normalize the labels while merging them.

# COMMAND ----------

dbutils.fs.mkdirs("/databricks/driver/Insurence_Risk_Factors/files/")
with open('/dbfs/databricks/driver/Insurence_Risk_Factors/files/replace_dict.csv', 'w') as f:
    f.write("""SMOKING,SMOKER
CAD,DISEASE
CEREBROVASCULAR_DISEASE,DISEASE
DIABETES,DISEASE
DISEASE_SYNDROME_DISORDER,DISEASE
HEART_DISEASE,DISEASE
HYPERLIPIDEMIA,DISEASE
HYPERTENSION,DISEASE
KIDNEY_DISEASE,DISEASE
MENTAL_HEALTH,DISEASE
OBESE,DISEASE
OBESITY,DISEASE
ONCOLOGICAL,DISEASE
OTHER_DISEASE,DISEASE
OVERWEIGHT,DISEASE
EKG_FINDINGS,DISEASE
IMAGINGFINDINGS,DISEASE
VS_FINDING,DISEASE
IMAGINGFINDINGS,DISEASE
DRUG_INGREDIENT,DRUG
DRUG_BRANDNAME,DRUG
MEDICATION,DRUG
SUBSTANCE_USE,SUBSTANCE
EMPLOYMENT,PROFESSION
MENTAL_HEALTH,PSYCHOLOGICAL_CONDITION
""")

# COMMAND ----------

chunk_merger = ChunkMergeApproach()\
    .setInputCols("ner_jsl_chunk", "ner_deid_chunk", "ner_risks_chunk", "ner_sdoh_chunk", "ner_posology_chunk", "chunk_age")\
    .setOutputCol('ner_chunk')\
    .setOrderingFeatures(["ChunkLength"])\
    .setSelectionStrategy("DiverseLonger")\
    .setReplaceDictResource('dbfs:/databricks/driver/Insurence_Risk_Factors/files/replace_dict.csv',"text", {"delimiter":","})

# COMMAND ----------

jsl_ner_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    risk_factors_ner,
    risk_factors_ner_converter,
    jsl_ner,
    jsl_ner_converter,
    sdoh_ner,
    sdoh_ner_converter,
    posology_ner,
    posology_ner_converter,
    deid_ner,
    deid_ner_converter,
    age_contextual_parser,
    age_chunk_converter,
    chunk_merger
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
jsl_ner_model = jsl_ner_pipeline.fit(empty_data)

lmodel= LightPipeline(jsl_ner_model)

# COMMAND ----------

# MAGIC %md
# MAGIC **Lets visualize the pipeline results by using Spark NLP Display `NerVisualizer`.**

# COMMAND ----------

from sparknlp_display import NerVisualizer

light_result = lmodel.fullAnnotate(df.select("text").take(1)[0]["text"])

visualiser = NerVisualizer()
ner_vis = visualiser.display(light_result[0], label_col='ner_chunk', document_col='document', return_html=True)
displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC We will transform the data using the pipeline and save the results as a delta table.

# COMMAND ----------

result = jsl_ner_model.transform(df.repartition(32))

# COMMAND ----------

delta_path='/FileStore/HLS/nlp/delta/jsl/'
 
result.write.format('delta').mode('overwrite').save(f'{delta_path}/Insurance_Risk_Factor/ner_result')
display(dbutils.fs.ls(f'{delta_path}/Insurance_Risk_Factor/ner_result'))

# COMMAND ----------

# MAGIC %md
# MAGIC Lets read the results from delta table and convert to a pandas dataframe.

# COMMAND ----------

result = spark.read.format('delta').load(f'{delta_path}/Insurance_Risk_Factor/ner_result')

result_pd = result.select(
    result.PATIENT_ID,
    result.file_name,
    F.explode(
        F.arrays_zip(
            result.ner_chunk.result, 
            result.ner_chunk.metadata)
    )
).select(
    result.PATIENT_ID,
    result.file_name,
    F.expr("col['0']").alias("chunk"),
    F.expr("col['1']['entity']").alias("label")
    ).toPandas()

result_pd

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will pivot the result dataframe and have a new dataframe that has the detected labels as columns.

# COMMAND ----------

df_slim = result_pd.groupby(['PATIENT_ID', 'file_name', 'label'])['chunk'].apply(list).reset_index()
df_slim_pivot = df_slim.pivot(['PATIENT_ID', 'file_name'], columns='label', values='chunk').fillna('').reset_index()
df_slim_pivot = df_slim_pivot.rename(columns={"file_name":"FILE"})

for i in df_slim_pivot.columns[2:]:
    df_slim_pivot[i] = df_slim_pivot[i].apply(lambda x: list(set([i.lower() for i in x])))

df_slim_pivot

# COMMAND ----------

df_slim_pivot.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gender Classifier
# MAGIC 
# MAGIC We will use Spark NLP Gender Classifier models to detect the genders of the patients and add them as a new Gender column to our dataframe.

# COMMAND ----------

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")\
    .setCaseSensitive(False)

gender_classifier = ClassifierDLModel.pretrained( 'classifierdl_gender_sbert', 'en', 'clinical/models') \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("gender")

gender_pipeline = Pipeline(stages=[document_assembler, sbert_embedder, gender_classifier])

gender_model = gender_pipeline.fit(empty_data)
gender_lmodel = LightPipeline(gender_model)

# COMMAND ----------

classes = [gender_lmodel.annotate(i)["gender"][0] for i in df_samples.text]
gender_df = pd.DataFrame({"PATIENT_ID": df_samples.PATIENT_ID, "GENDER":classes})
df_slim_pivot = pd.merge(df_slim_pivot, gender_df, on="PATIENT_ID")
df_slim_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ## ICD
# MAGIC 
# MAGIC We will find the ICD10CM codes of the detected `DISEASE` entities by using Spark NLP Sentence Entity Resolver Model and add as a new column.

# COMMAND ----------

documentAssemblerResolver = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("ner_chunks")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
    .setInputCols(["ner_chunks"])\
    .setOutputCol("sentence_embeddings")\
    .setCaseSensitive(False)
    
icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models") \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("icd10cm_code")\
    .setDistanceFunction("EUCLIDEAN")

icd_pipelineModel = PipelineModel(stages=[
            documentAssemblerResolver,
            sbert_embedder,
            icd_resolver
            ])

icd10_lp = LightPipeline(icd_pipelineModel)

# COMMAND ----------

# MAGIC %%time 
# MAGIC 
# MAGIC df_slim_pivot['DISEASE_ICD'] = df_slim_pivot['DISEASE'].apply(lambda x : [get_resolver_code(icd10_lp, disease_list, "icd10cm_code") for disease_list in x] if len(x)>0 else [])
# MAGIC df_slim_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ## RxNorm 
# MAGIC 
# MAGIC We will find the RxNorm codes of the detected `DRUG` entities by using Spark NLP Sentence Entity Resolver Model and add as a new column.

# COMMAND ----------

documentAssemblerResolver = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("ner_chunks")

sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
    .setInputCols(["ner_chunks"])\
    .setOutputCol("sentence_embeddings")\
    .setCaseSensitive(False)
    
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models") \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("rxnorm_code")\
    .setDistanceFunction("EUCLIDEAN")

rxnorm_pipelineModel = PipelineModel(
    stages = [
        documentAssemblerResolver,
        sbert_embedder,
        rxnorm_resolver])

rxnorm_lp = LightPipeline(rxnorm_pipelineModel)

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC df_slim_pivot['DRUG_RXNORM'] = df_slim_pivot['DRUG'].apply(lambda x : [get_resolver_code(rxnorm_lp, drug_list, "rxnorm_code") for drug_list in x] if len(x)>0 else [])
# MAGIC df_slim_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ## RxNorm-Drug Actions Mapper
# MAGIC 
# MAGIC Now we will get the actions of the drugs to check if their actions are in risky category.

# COMMAND ----------

print(df_slim_pivot.columns)

# COMMAND ----------

documentAssemblerMapper = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("rxnorm_code")

action_mapper = ChunkMapperModel\
    .pretrained("rxnorm_action_treatment_mapper", "en", "clinical/models")\
    .setInputCols(["rxnorm_code"])\
    .setOutputCol("action")\
    .setRels(["action"])

mapper_pipeline = PipelineModel(
    stages = [
        documentAssemblerMapper,
        action_mapper,
    ])

mapper_lp = LightPipeline(mapper_pipeline)

# COMMAND ----------

df_slim_pivot['DRUG_ACTIONS'] = df_slim_pivot['DRUG_RXNORM'].apply(lambda x : list(set([get_treatment_action(a) for a in x])))
df_slim_pivot['DRUG_ACTIONS'] = df_slim_pivot.DRUG_ACTIONS.apply(lambda x : [a for a in x if a[1] != 'NONE'])
df_slim_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ## Family Health History
# MAGIC 
# MAGIC We can get the family health histories of the patients by using Spark NLP Assertion Status Detection models.

# COMMAND ----------

chunk_merger_assertion = ChunkMergeApproach()\
    .setInputCols("ner_jsl_chunk", 'ner_risks_chunk', "ner_sdoh_chunk")\
    .setOutputCol('assertion_chunk')\
    .setOrderingFeatures(["ChunkLength"])\
    .setSelectionStrategy("DiverseLonger")\
    .setReplaceDictResource('dbfs:/databricks/driver/Insurence_Risk_Factors/files/replace_dict.csv',"text", {"delimiter":","})

chunk_filterer = ChunkFilterer()\
    .setInputCols("sentence","assertion_chunk")\
    .setOutputCol("filtered_chunk")\
    .setFilterEntity("entity")\
    .setWhiteList(["DISEASE"])

clinical_assertion = AssertionDLModel.pretrained("assertion_jsl_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "filtered_chunk", "embeddings"]) \
    .setOutputCol("assertion")

assertion_filterer = AssertionFilterer()\
    .setInputCols("sentence","filtered_chunk","assertion")\
    .setOutputCol("assertion_filtered")\
    .setWhiteList(["Family"])


family_assertion_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    risk_factors_ner,
    risk_factors_ner_converter,
    jsl_ner,
    jsl_ner_converter,
    sdoh_ner,
    sdoh_ner_converter,
    chunk_merger_assertion,
    chunk_filterer,
    clinical_assertion,
    assertion_filterer
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
family_assertion_model = family_assertion_pipeline.fit(empty_data)

family_assertion_lmodel= LightPipeline(family_assertion_model)

# COMMAND ----------

# MAGIC %%time 
# MAGIC 
# MAGIC family_diseases = [list(set(family_assertion_lmodel.annotate(i)["assertion_filtered"])) for i in df_samples.text]
# MAGIC family_disease_df = pd.DataFrame({"PATIENT_ID": df_samples.PATIENT_ID, "FAMILY_DISEASE":family_diseases})
# MAGIC family_disease_df.FAMILY_DISEASE = family_disease_df.FAMILY_DISEASE.apply(lambda x: list(set([i.lower() for i in x])) if len(x)>0 else [])
# MAGIC df_slim_pivot = pd.merge(df_slim_pivot, family_disease_df, on="PATIENT_ID")
# MAGIC df_slim_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC Lets add the ICD-10-CM codes of the diseases that family members have.

# COMMAND ----------

# MAGIC %%time 
# MAGIC 
# MAGIC df_slim_pivot['FAMILY_DISEASE_ICD'] = df_slim_pivot['FAMILY_DISEASE'].apply(lambda x : [get_resolver_code(icd10_lp, a, "icd10cm_code") for a in x] if len(x)>0 else [])
# MAGIC df_slim_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ## Status of Alcohol, Tobacco and Substance Behaviours
# MAGIC 
# MAGIC We will check the alcohol, tobacco and substance behaviours of the patients.

# COMMAND ----------

chunk_filterer_behaviour = ChunkFilterer()\
    .setInputCols("sentence","assertion_chunk")\
    .setOutputCol("filtered_chunk")\
    .setFilterEntity("entity")\
    .setWhiteList(["ALCOHOL", "SUBSTANCE", "SMOKER"])

clinical_assertion = AssertionDLModel.pretrained("assertion_jsl_augmented", "en", "clinical/models") \
    .setInputCols(["sentence", "filtered_chunk", "embeddings"]) \
    .setOutputCol("assertion")


behaviour_assertion_pipeline = Pipeline(stages=[
    documentAssembler, 
    sentenceDetector,
    tokenizer,
    word_embeddings,
    risk_factors_ner,
    risk_factors_ner_converter,
    jsl_ner,
    jsl_ner_converter,
    sdoh_ner,
    sdoh_ner_converter,
    chunk_merger_assertion,
    chunk_filterer_behaviour,
    clinical_assertion
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
behaviour_assertion_model = behaviour_assertion_pipeline.fit(empty_data)

behaviour_assertion_lmodel= LightPipeline(behaviour_assertion_model)

# COMMAND ----------

# MAGIC %%time
# MAGIC ann_result = behaviour_assertion_lmodel.fullAnnotate(df_samples.text.to_list())

# COMMAND ----------

alcohol_status = []
substance_status = []
tobacco_status = []

for k in ann_result:
    label = []
    assertion = []
    
    for i,j in list(zip(k["filtered_chunk"], k["assertion"])):
        label.append(i.metadata["entity"])
        assertion.append(j.result)

    tmp_df = pd.DataFrame({"label":label, "assertion":assertion})
    tmp_df = tmp_df.groupby("label")["assertion"].apply(list).reset_index()
    tmp_df["behaviour"] = tmp_df.assertion.apply(lambda x: True if "Present" in x else False)
    tmp_filtered = tmp_df[tmp_df.behaviour==True]

    if "ALCOHOL" in tmp_filtered.label.to_list():
        alcohol_status.append(True)
    else: alcohol_status.append(False)

    if "SMOKER" in tmp_filtered.label.to_list():
        tobacco_status.append(True)
    else: tobacco_status.append(False)

    if "SUBSTANCE" in tmp_filtered.label.to_list():
        substance_status.append(True)
    else: substance_status.append(False)

behaviour_df = pd.DataFrame({"PATIENT_ID":df_samples.PATIENT_ID, "ALCOHOL_STATUS":alcohol_status, "TOBACCO_STATUS":tobacco_status, "SUBSTANCE_STATUS":substance_status})
df_slim_pivot = df_slim_pivot.merge(behaviour_df, on = "PATIENT_ID")
df_slim_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ## Social Determinants of Health Classification Models
# MAGIC 
# MAGIC Another way to detect the alcohol, substance and tobacco behviours of the patients is using the SDOH classification models in Spark NLP.
# MAGIC 
# MAGIC Models:
# MAGIC 
# MAGIC - [genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli](https://nlp.johnsnowlabs.com/2023/01/14/genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli_en.html): 
# MAGIC   + *Present*
# MAGIC   + *Never*
# MAGIC   + *None* 
# MAGIC - [genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli](https://nlp.johnsnowlabs.com/2023/01/14/genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli_en.html): 
# MAGIC   + *Present*
# MAGIC   + *Past*
# MAGIC   + *Never*
# MAGIC   + *None*
# MAGIC - [genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli](https://nlp.johnsnowlabs.com/2023/01/14/genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli_en.html): 
# MAGIC   + *Present:* if the patient was a current consumer of substance or the patient was a consumer in the past and had quit or if the patient had never consumed substance. 
# MAGIC   + *None:* if there was no related text.

# COMMAND ----------

documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sbertEmbedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

features_asm = FeaturesAssembler()\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("features")

# substance usage binary classifier: Present, None
substance_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("substance_status")

# alcohol usage classifier: Present, Never, None
alcohol_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("alcohol_status")

# tobacco usage classifier: Present, Past, Never, None
tobacco_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("tobacco_status")

classifier_pipeline = Pipeline(stages=[
    documentAssembler,
    sbertEmbedder,
    features_asm,
    substance_classifier,
    alcohol_classifier,
    tobacco_classifier
])

empty_data = spark.createDataFrame([[""]]).toDF("text")
clf_model = classifier_pipeline.fit(empty_data)
clf_light_model = LightPipeline(clf_model)

# COMMAND ----------

text_list = [
    "He drinks alcohol ten to twelve drinks a week, but does not drink five days a week and then will binge drink. He smokes one and a half pack a day for 15 years.", # text in PATIENT_ID: #66148
    "As mentioned before, the patient's toxicology screen was positive for morphine, methadone, and marijuana.", # text in PATIENT_ID: #24168
    "Ethanol, tobacco, or drugs; he smoked 2 packs per day for 40 years, but quit in 1996. He occasionally has a beer, but denies any continuous use of alcohol. He denies any illicit drug use." # text in PATIENT_ID: #96400
    
]

lmodel_behaviour = clf_light_model.fullAnnotate(text_list)
substance_status = [j.result for i in lmodel_behaviour for j in i["substance_status"]]
tobacco_status = [j.result for i in lmodel_behaviour for j in i["tobacco_status"]]
alcohol_status = [j.result for i in lmodel_behaviour for j in i["alcohol_status"]]

behaviour_clf_df = pd.DataFrame({"text":text_list, "alcohol_status":alcohol_status, "substance_status":substance_status, "tobacco_status":tobacco_status})
behaviour_clf_df

# COMMAND ----------

# MAGIC %md
# MAGIC # ANALYSIS
# MAGIC 
# MAGIC It is time for analysis!
# MAGIC 
# MAGIC Since the dataframe is a small one, it is not enough for making a good analysis. So we did the same steps for a larger one and we will use that dataframe for making analysis.

# COMMAND ----------

#downloading the sample dataset
! wget -q https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/healthcare-nlp/data/insurance_risk_factors_df.pickle -P /dbfs/databricks/driver/Insurence_Risk_Factors/data

# COMMAND ----------

insurance_risk_df = pd.read_pickle("/dbfs/databricks/driver/Insurence_Risk_Factors/data/insurance_risk_factors_df.pickle")
insurance_risk_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## RACE ETHNICITY-BEHAVIOR ANALYSIS
# MAGIC 
# MAGIC Lets check the relations between race ethnicities and behavious of patients.

# COMMAND ----------

insurance_risk_df.RACE_ETHNICITY.value_counts()

# COMMAND ----------

whites = insurance_risk_df[insurance_risk_df.RACE_ETHNICITY.astype(str).str.contains("white")].copy()
blacks = insurance_risk_df[(insurance_risk_df.RACE_ETHNICITY.astype(str).str.contains("black")) | (insurance_risk_df.RACE_ETHNICITY.astype(str).str.contains("africa"))].copy()
whites["RACE"] = "WHITE"
blacks["RACE"] = "BLACK"
race_df = pd.concat([whites, blacks]).reset_index(drop=True)

race_behaviour_df = race_df[["RACE", "SUBSTANCE_STATUS", "ALCOHOL_STATUS", "TOBACCO_STATUS"]]
race_behaviour_df = race_behaviour_df.groupby("RACE").sum().reset_index()
race_behaviour_df = race_behaviour_df.merge(pd.DataFrame(race_df.RACE.value_counts().reset_index()).rename(columns = {"RACE":"TOTAL", "index":"RACE"}), on = "RACE")
race_behaviour_df

# COMMAND ----------

# plot race_behaviour dataframe.
fig, ax = plt.subplots(figsize=(12,6) ,dpi=90)
plt.title('Race Ethnicity VS Behavior Counts', size=15)
race_behaviour_df.plot(x="RACE", y=["TOTAL", "SUBSTANCE_STATUS", "ALCOHOL_STATUS", "TOBACCO_STATUS"], kind="bar", ax=ax)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## GENDER-BASED ANALYSIS
# MAGIC 
# MAGIC Lets make some gender based analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### GENDER-BEHAVIOUR ANALYSIS

# COMMAND ----------

insurance_risk_df.GENDER.value_counts()

# COMMAND ----------

gender_df = insurance_risk_df[insurance_risk_df.GENDER != "Unknown"].copy()   # drop Unknown 
gender_behaviour_df = gender_df[["GENDER", "SUBSTANCE_STATUS", "ALCOHOL_STATUS", "TOBACCO_STATUS"]]
gender_behaviour_df = gender_behaviour_df.groupby("GENDER").sum().reset_index()
gender_behaviour_df = gender_behaviour_df.merge(pd.DataFrame(gender_df.GENDER.value_counts().reset_index()).rename(columns = {"GENDER":"TOTAL", "index":"GENDER"}), on = "GENDER")
gender_behaviour_df

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12,6) ,dpi=90)
plt.title('Gender VS Behaviour Counts', size=15)
gender_behaviour_df.plot(x="GENDER", y=["TOTAL", "TOBACCO_STATUS", "ALCOHOL_STATUS", "SUBSTANCE_STATUS"], kind="bar", ax=ax)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### OBESITY-GENDER-BEHAVIOUR ANALYSIS

# COMMAND ----------

gender_df["DISEASE_NORMALIZED"] = get_normalized_name(gender_df, "DISEASE_ICD")
obesity_df = gender_df[gender_df.DISEASE_NORMALIZED.astype(str).str.contains("obesity")].reset_index(drop=True)

obesity_behaviour_df = obesity_df[["GENDER", "SUBSTANCE_STATUS", "ALCOHOL_STATUS", "TOBACCO_STATUS"]]
obesity_behaviour_df = obesity_behaviour_df.groupby("GENDER").sum().reset_index()
obesity_behaviour_df = obesity_behaviour_df.merge(pd.DataFrame(obesity_df.GENDER.value_counts().reset_index()).rename(columns = {"GENDER":"TOTAL", "index":"GENDER"}), on = "GENDER")

fig, ax = plt.subplots(figsize=(12,6) ,dpi=90)
plt.title('Gender VS Behaviour Counts of Patients Have Obesity', size=15)
obesity_behaviour_df.plot(x="GENDER", y=["TOTAL", "TOBACCO_STATUS", "ALCOHOL_STATUS", "SUBSTANCE_STATUS"], kind="bar", ax=ax)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## DISEASE ANALYSIS
# MAGIC 
# MAGIC We will analyze the disease based extractions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most Common Diseases
# MAGIC 
# MAGIC Lets check the most common diseases that patients have.

# COMMAND ----------

counts = get_occurence(insurance_risk_df, "DISEASE", 1)
top_20_disease = [(i[0][0], i[1]) for i in counts.most_common()[:20]]
top_20_disease_df = pd.DataFrame(top_20_disease, columns = ["DISEASE", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Diseases', size=15)
top_20_disease_df.plot(x="DISEASE", y=["COUNT"], kind="bar", ax=ax, color="darkred")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most Common Diseases With Normalized Names 
# MAGIC 
# MAGIC Disease extractions can have different formats, so when we will normalize the diseases by using the ICD-10-CM codes and check the most common normalized diseases and see the results will be changed.

# COMMAND ----------

insurance_risk_df["DISEASE_NORMALIZED"] = get_normalized_name(insurance_risk_df, "DISEASE_ICD")
insurance_risk_df["DISEASE_NORMALIZED"] = insurance_risk_df["DISEASE_NORMALIZED"].apply(lambda x: list(set(x)))

counts = get_occurence(insurance_risk_df, "DISEASE_NORMALIZED", 1)
top_20_disease = [(i[0][0], i[1]) for i in counts.most_common()[:20]]
top_20_disease_df = pd.DataFrame(top_20_disease, columns = ["NORMALIZED_DISEASE_NAMES", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Diseases With Normalized Names', size=15)
top_20_disease_df.plot(x="NORMALIZED_DISEASE_NAMES", y=["COUNT"], kind="bar", ax=ax, color="cadetblue")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **As you can see, some of the disease orders have been changed after normalization of the disease column.**
# MAGIC 
# MAGIC *NOTES:*
# MAGIC 
# MAGIC + Growth hormone deficiency (GHD) is a rare and treatable condition that causes short height in children and metabolic issues in adults. 
# MAGIC + FH is a genetic condition that causes high cholesterol. Familial hypercholesterolemia (FH) is a genetic disorder that affects about 1 in 250 people and increases the likelihood of having coronary heart disease at a younger age.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most Common Disease Pairs
# MAGIC 
# MAGIC Lets check which diseases seen together in the same patient clinical note.

# COMMAND ----------

counts = get_occurence(insurance_risk_df, "DISEASE", 2)
top_20_disease_pairs = [(i[0], i[1]) for i in counts.most_common()[:20]]
top_20_disease_pairs_df = pd.DataFrame(top_20_disease_pairs, columns = ["DISEASE_PAIRS", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Disease Pairs', size=15)
top_20_disease_pairs_df.plot(x="DISEASE_PAIRS", y=["COUNT"], kind="bar", ax=ax, color="darkred")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most Common Disease Pairs With Normalized Names 
# MAGIC 
# MAGIC Lets check normalized disease pairs.

# COMMAND ----------

counts = get_occurence(insurance_risk_df, "DISEASE_NORMALIZED", 2)
top_20_disease_pairs = [(i[0], i[1]) for i in counts.most_common()[:20]]
top_20_disease_pairs_df = pd.DataFrame(top_20_disease_pairs, columns = ["NORMALIZED_DISEASE_PAIRS", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Disease Pairs With Normalized Names', size=15)
top_20_disease_pairs_df.plot(x="NORMALIZED_DISEASE_PAIRS", y=["COUNT"], kind="bar", ax=ax, color="cadetblue")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What kind of diseases do patients who drink alcohol have?

# COMMAND ----------

alcohol_df = insurance_risk_df[insurance_risk_df["ALCOHOL_STATUS"]==True]

counts = get_occurence(alcohol_df, "DISEASE_NORMALIZED", 1)
top_20_disease = [(i[0][0], i[1]) for i in counts.most_common()[:20]]
top_20_disease_df = pd.DataFrame(top_20_disease, columns = ["NORMALIZED_DISEASE_NAMES", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Diseases Of The Patients Use Alcohol', size=15)
top_20_disease_df.plot(x="NORMALIZED_DISEASE_NAMES", y=["COUNT"], kind="bar", ax=ax, color="cadetblue")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What kind of diseases do smokers have?

# COMMAND ----------

tobacco_df = insurance_risk_df[insurance_risk_df["TOBACCO_STATUS"]==True]

counts = get_occurence(tobacco_df, "DISEASE_NORMALIZED", 1)
top_20_disease = [(i[0][0], i[1]) for i in counts.most_common()[:20]]
top_20_disease_df = pd.DataFrame(top_20_disease, columns = ["NORMALIZED_DISEASE_NAMES", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Diseases Of The Patients Use Tobacco', size=15)
top_20_disease_df.plot(x="NORMALIZED_DISEASE_NAMES", y=["COUNT"], kind="bar", ax=ax, color="cadetblue")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## DRUG ANALYSIS
# MAGIC 
# MAGIC We will make analysis on the drug based extractions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most Common Used Drugs

# COMMAND ----------

counts = get_occurence(insurance_risk_df, "DRUG", 1)
top_20_drug = [(i[0][0], i[1]) for i in counts.most_common()[:20]]
top_20_drug_df = pd.DataFrame(top_20_drug, columns = ["DRUG", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Used Drugs', size=15)
top_20_drug_df.plot(x="DRUG", y=["COUNT"], kind="bar", ax=ax, color="royalblue")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most Common Used Drugs With Normalized Names 

# COMMAND ----------

insurance_risk_df["DRUG_NORMALIZED"] = get_normalized_name(insurance_risk_df, "DRUG_RXNORM")
insurance_risk_df["DRUG_NORMALIZED"] = insurance_risk_df["DRUG_NORMALIZED"].apply(lambda x: list(set(x)))    # if multiple drugs have the same actions

counts = get_occurence(insurance_risk_df, "DRUG_NORMALIZED", 1)
top_20_drug = [(i[0][0], i[1]) for i in counts.most_common()[:20]]
top_20_drug_df = pd.DataFrame(top_20_drug, columns = ["NORMALIZED_DRUG_NAMES", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Used Drugs With Normalized Names', size=15)
top_20_drug_df.plot(x="NORMALIZED_DRUG_NAMES", y=["COUNT"], kind="bar", ax=ax, color="darkblue")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most Common Used Drug Pairs
# MAGIC 
# MAGIC Lets check which drugs used together by the patients.

# COMMAND ----------

counts = get_occurence(insurance_risk_df, "DRUG", 2)
top_20_drug_pairs = [(i[0], i[1]) for i in counts.most_common()[:20]]
top_20_drug_pairs_df = pd.DataFrame(top_20_drug_pairs, columns = ["DRUG_PAIRS", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Used Drug Pairs', size=15)
top_20_drug_pairs_df.plot(x="DRUG_PAIRS", y=["COUNT"], kind="bar", ax=ax, color="skyblue")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most Common Used Drug Pairs With Normalized Names 

# COMMAND ----------

counts = get_occurence(insurance_risk_df, "DRUG_NORMALIZED", 2)
top_20_drug_pairs = [(i[0], i[1]) for i in counts.most_common()[:20]]
top_20_drug_pairs_df = pd.DataFrame(top_20_drug_pairs, columns = ["DRUG_NORMALIZED", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Used Drug Pairs With Normalized Names', size=15)
top_20_drug_pairs_df.plot(x="DRUG_NORMALIZED", y=["COUNT"], kind="bar", ax=ax, color="teal")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most Common Actions of Drugs
# MAGIC 
# MAGIC We will check the most comoon actions of the drugs.

# COMMAND ----------

insurance_risk_df["DRUG_ACTIONS_NORMALIZED"] = get_normalized_name(insurance_risk_df, "DRUG_ACTIONS")
insurance_risk_df["DRUG_ACTIONS_NORMALIZED"] = insurance_risk_df["DRUG_ACTIONS_NORMALIZED"].apply(lambda x: list(set(x)))

counts = get_occurence(insurance_risk_df, "DRUG_ACTIONS_NORMALIZED", 1)
top_20_actions = [(i[0][0], i[1]) for i in counts.most_common()[:20]]
top_20_actions_df = pd.DataFrame(top_20_actions, columns = ["DRUG_ACTIONS", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Actions of Drugs', size=15)
top_20_actions_df.plot(x="DRUG_ACTIONS", y=["COUNT"], kind="bar", ax=ax, color="palevioletred")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What kind of diseases do patients who use `Analgesic` drugs have?
# MAGIC 
# MAGIC As you can see, the most common drug action is `Analgesic`. Lets check what kind of diseases the patients who use `Analgesic` drugs have. 

# COMMAND ----------

analgesic_df = insurance_risk_df[insurance_risk_df["DRUG_ACTIONS_NORMALIZED"].astype(str).str.contains("Analgesic")]
counts = get_occurence(analgesic_df, "DISEASE_NORMALIZED", 1)
top_20_actions = [(i[0][0], i[1]) for i in counts.most_common()[:20]]
top_20_actions_df = pd.DataFrame(top_20_actions, columns = ["DISEASE", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Diseases of The Patients Use Analgesic Drugs', size=15)
top_20_actions_df.plot(x="DISEASE", y=["COUNT"], kind="bar", ax=ax, color="goldenrod")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What are the most common drug-action pairs?

# COMMAND ----------

counts = get_occurence(insurance_risk_df, "DRUG_ACTIONS", 1)
top_20_drug_action_pairs = [(i[0], i[1]) for i in counts.most_common()[:20]]
top_20_drug_action_pairs_df = pd.DataFrame(top_20_drug_action_pairs, columns = ["DRUG_ACTION", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Drug-Action Pairs', size=15)
top_20_drug_action_pairs_df.plot(x="DRUG_ACTION", y=["COUNT"], kind="bar", ax=ax, color="rebeccapurple")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## HYPERTENSION ANALYSIS
# MAGIC 
# MAGIC We found that *hypertension* is the most common disease in the patient notes. Now we will create a *hypertension* sub-dataframe on the `DISEASE_NORMALIZED` column and make some analysis on this dataframe.

# COMMAND ----------

hypertension_df = insurance_risk_df.loc[insurance_risk_df.DISEASE_NORMALIZED.apply(lambda x: x if "hypertension" in x else None).dropna().index].reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### What are the genders of the patients have `hypertension`?

# COMMAND ----------

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Genders of Patients Have Hypertension', size=15)
hypertension_df[hypertension_df.GENDER != "Unknown"].GENDER.value_counts().plot(kind="bar", ax=ax, color=["crimson", "blue"])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### What diseases do family members of the `hypertension` patients have?

# COMMAND ----------

counts = get_occurence(hypertension_df, "FAMILY_DISEASE", 1)
top_20_disease = [(i[0][0], i[1]) for i in counts.most_common()[:10]]
top_20_disease_df = pd.DataFrame(top_20_disease, columns = ["FAMILY_DISEASE", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Family Diseases Of Patients Have Hypertension', size=15)
top_20_disease_df.plot(x="FAMILY_DISEASE", y=["COUNT"], kind="bar", ax=ax, color="cadetblue")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### What `drugs` do `hypertension` patients use?

# COMMAND ----------

counts = get_occurence(hypertension_df, "DRUG_NORMALIZED", 1)
top_20_disease = [(i[0][0], i[1]) for i in counts.most_common()[:10]]
top_20_disease_df = pd.DataFrame(top_20_disease, columns = ["DRUG", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Used Drugs Of Patients Have Hypertension', size=15)
top_20_disease_df.plot(x="DRUG", y=["COUNT"], kind="bar", ax=ax, color="seagreen")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### What are the most common `actions` of the drugs that `hypertension` patients use?

# COMMAND ----------

counts = get_occurence(hypertension_df, "DRUG_ACTIONS_NORMALIZED", 1)
top_20_disease = [(i[0][0], i[1]) for i in counts.most_common()[:10]]
top_20_disease_df = pd.DataFrame(top_20_disease, columns = ["DRUG_ACTIONS", "COUNT"])

fig, ax = plt.subplots(figsize=(20,6) ,dpi=90)
plt.title('Most Common Drug Actions of Drugs Used by Patients Have Hypertension', size=15)
top_20_disease_df.plot(x="DRUG_ACTIONS", y=["COUNT"], kind="bar", ax=ax, color="palevioletred")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Behaviours of Hypertension Patients

# COMMAND ----------

hypertension_behaviour_df = hypertension_df[["SUBSTANCE_STATUS", "ALCOHOL_STATUS", "TOBACCO_STATUS"]]

fig, ax = plt.subplots(figsize=(12,6) ,dpi=90)
plt.title('Behaviours of Hypertension Patients', size=15)
sns.countplot(x="variable", hue="value", data=pd.melt(hypertension_behaviour_df), palette=["darkgreen", "firebrick"])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.010))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |MatPlotLib | | https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE | https://github.com/matplotlib/matplotlib|
# MAGIC |Seaborn |BSD 3-Clause License | https://github.com/seaborn/seaborn/blob/master/LICENSE | https://github.com/seaborn/seaborn/|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC 
# MAGIC 
# MAGIC |Author|
# MAGIC |-|
# MAGIC |Databricks Inc.|
# MAGIC |John Snow Labs Inc.|

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disclaimers
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
