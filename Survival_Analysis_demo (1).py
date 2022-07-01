# Databricks notebook source
# MAGIC %md
# MAGIC #Introduction
# MAGIC 
# MAGIC -**Defination:** Survival analysis is a set of statistical approaches used to find out the time it takes for an event of interest to occur. Survival analysis is used to study the time until some event of interest (often referred to as death) occurs. Time could be measured in years, months, weeks, days, etc. The event of interest could be anything of interest. It could be an actual death, a birth, a retirement etc.
# MAGIC 
# MAGIC -**Survival analysis is used in a variety of field such as:**
# MAGIC 
# MAGIC Cancer studies for patients survival time analyses,
# MAGIC Sociology for “event-history analysis",
# MAGIC In Engineering for “failure-time analysis”,
# MAGIC Time until product failure,
# MAGIC Time until a warranty claim.
# MAGIC Time until a process reaches a critical level,
# MAGIC Time from initial sales contact to a sale,
# MAGIC Time from employee hire to either termination or quit,
# MAGIC Time from a salesperson hire to their first sale
# MAGIC 
# MAGIC -**Main terminologies in survival analysis:**
# MAGIC 
# MAGIC 1.**Survival function**: Probability that instance will survive longer than time t
# MAGIC 
# MAGIC 2.**Cumulative density function**:Probability that event occurs before time t 
# MAGIC 
# MAGIC 3.**Hazard function**: Probability that instance survived till time t and then event happened.
# MAGIC 
# MAGIC 4.**Baseline function**: This is nothing but which feature combination have maximum probablity of event happening. E.g If payment_method == 'ATM' have maximum number 
# MAGIC   of churn customers then baseline model gets created using payment_method == 'ATM'   
# MAGIC 
# MAGIC 5.**Censorship**: Censorship allows you to measure lifetimes for the population who haven’t experienced the event of interest yet.There are different types of Censorship done in Survival Analysis as explained below:
# MAGIC 
# MAGIC *Right Censoring*: This happens when the subject enters at t=0 i.e at the start of the study and terminates before the event of interest occurs. This can be either not experiencing the event of interest during the study, i.e they lived longer than the duration of the study, or could not be a part of the study completely and left early without experiencing the event of interest, i.e they left and we could not study them any longer.
# MAGIC 
# MAGIC *Left Censoring*: This happens when the birth event wasn’t observed. Another concept known as Length-Biased Sampling should also be mentioned here. This type of sampling occurs when the goal of the study is to perform analysis on the people/subjects who already experienced the event and we wish to see whether they will experience it again. The lifelines package has support for left-censored datasets by adding the keyword left_censoring=True. Note that by default, it is set to False.:
# MAGIC 
# MAGIC *Interval Censoring*: This happens when the follow-up period, i.e time between observation, is not continuous. This can be weekly, monthly, quarterly, etc.
# MAGIC 
# MAGIC *Left Truncation*: It is referred to as late entry. The subjects may have experienced the event of interest before entering the study. There is an argument named ‘entry’ that specifies the duration between birth and entering the study. If we fill in the truncated region then it will make us overconfident about what occurs in the early period after diagnosis. That’s why we truncate them[9].
# MAGIC 
# MAGIC *In short, subjects who have not experienced the event of interest during the study period are right-censored and subjects whose birth has not been seen are left-censored. Survival Analysis was developed to mainly solve the problem of right-censoring.*
# MAGIC 
# MAGIC  
# MAGIC  
# MAGIC 
# MAGIC 
# MAGIC -**3 Types of survival statistical method:**
# MAGIC 
# MAGIC 1.Non parametric method: Eg. Kaplan–Meier estimator
# MAGIC 
# MAGIC 2.Semi parametric method: Eg.Cox proportional hazard model
# MAGIC 
# MAGIC 3.Parametric method: Accelerated Failure Time Model(AFT)
# MAGIC 
# MAGIC -**Evaluation metrix:**
# MAGIC 
# MAGIC 1.Concordance index (C-index)
# MAGIC if I event happened before j and c[i] < c[j] then its right prediction.
# MAGIC C-index takes into consideration only non censored events.it skips censored events
# MAGIC C index is nothing but weighted average of area under curve for each time interval
# MAGIC 
# MAGIC 2.Brier score:when censored events are more then need to reduce weight of these events in these cases brier score used.
# MAGIC 
# MAGIC 3.Mean Absolute Error: It ignores censored events and considers instances for which actual event occurred

# COMMAND ----------

# MAGIC %md
# MAGIC # Problem statement:
# MAGIC 
# MAGIC **For Frequent sauda customers find if next trade will occur in K input days.**

# COMMAND ----------

#Package used for survival analysis models
#pip install lifelines

# COMMAND ----------

# DBTITLE 1,Importing packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql.functions import expr,datediff,col,when
import matplotlib.pyplot as plt


from lifelines import KaplanMeierFitter,WeibullFitter,ExponentialFitter,LogNormalFitter,LogLogisticFitter,WeibullAFTFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import pairwise_logrank_test
from lifelines.fitters.coxph_fitter import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from mlflow import sklearn
import shutil

# COMMAND ----------

# DBTITLE 1,Data creation
#For training considering all frequent DTUs for 01-Nov-2021. 
#Considering 90 while calculating target as we have categories till Quarter

df = sqlContext.sql(f""" select distinct party_code,sauda_date,average_gap_last_four_trades_weeks,average_gap_last_four_trades_label,cumulative_trade_days, lead(sauda_date) over (partition by party_code order by sauda_date) as next_date  from ora_dormancy.user_daily_trade_summary 
where cumulative_trade_days > 3 and sauda_date >= '2021-11-01'  """)
df = df.filter(col('sauda_date') == '2021-11-01')
df = df.withColumn("tenure",datediff(col("next_date"),col("sauda_date")))
df1 = df['party_code','average_gap_last_four_trades_weeks','cumulative_trade_days','tenure','average_gap_last_four_trades_label']
new_column = when(col("tenure") <= 90, 1).otherwise(0)
df1 = df1.withColumn("target", new_column)
#df1 = df1.withColumn('tenure_week',expr("tenure/7"))
#df1=df1.drop('tenure')

df1.createOrReplaceTempView("survival_analysis1")
df_sn = spark.read.table('online_engine.client_kyc')
df_sn.createOrReplaceTempView("SN_ClientKYC")


#Adding categorical/demographic variables

df3 = sqlContext.sql(f""" select * from ((select * from survival_analysis1) a  left join
                         (select distinct party_code as party_code1, gender,riskcategory,incomedetails,occupation,City from SN_ClientKYC) b
                         on a.party_code = b.party_code1)""")


df2=df3.toPandas()   
#Replacing null values. Null values in tenure means no next trade.
#df2["tenure_week"].fillna(14, inplace = True)
df2["tenure"].fillna(100, inplace = True)
df2.drop(['party_code', 'party_code1','cumulative_trade_days','average_gap_last_four_trades_weeks'], axis=1, inplace=True)
df1.unpersist()
display(df2)

# COMMAND ----------

df2.shape

# COMMAND ----------

df2['target'].value_counts()

# COMMAND ----------

df2.isnull().sum()

# COMMAND ----------

df2['average_gap_last_four_trades_label'].value_counts()

# COMMAND ----------

# DBTITLE 1,Data manipulation
#Converting city to tier
tier_info1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/Indian_Cities_by_Tier_I_and_II.csv')
tier_info=tier_info1.toPandas()
tier_info.dropna(subset=['Tier'], inplace=True)
tier_mapper = tier_info[['City', 'Tier']].set_index('City').to_dict()['Tier']    
def tier_mapping(city):
    try: return tier_mapper[city.capitalize()]
    except: return 'Tier-III'
df2['Tier'] = df2['City'].map(tier_mapping)
df2 = df2.drop(['City'], axis = 1)
#Missing values
df2['occupation'].fillna('no_data', inplace=True)
df2['riskcategory'].fillna('other', inplace=True)  
    #Calculating income buckets
def f1(row):
    if row =='BELOW 1 LAC':
        val = 1
    elif row == '1-5 LAC':
        val = 2
    elif row == '5-10 LAC':
        val = 3
    elif row == '10-25 LAC':
        val = 4
    elif row == '>25 LAC':
        val = 5
    else:
        val = 0
    return val
df2['new_incomedetails'] = df2['incomedetails'].apply(f1)

df2["gender"].replace({"f": "F"}, inplace=True)
def f2(row):
    if row == 'F':
        val = 1
    else:
        val = 0
    return val
df2['gender_f'] = df2['gender'].apply(f2)

def f3(row):
    if row == 'HIGH':
         val = 2
    elif row == 'MEDIUM':
         val = 1
    else:
         val=0
    return val
df2['riskcategory_new'] = df2['riskcategory'].apply(f3)

def f4(row):
    if row == 'Week':
         val = 7
    elif row == 'TwoWeeks':
         val = 15
    elif row == 'Month':
         val = 30
    elif row == 'Quarter':
         val = 90          
    else:
         val=100
    return val
df2['average_gap_last_four_trades_label_new'] = df2['average_gap_last_four_trades_label'].apply(f4)


df2.drop(['gender', 'riskcategory','incomedetails','average_gap_last_four_trades_label'], axis=1, inplace=True)
display(df2)

# COMMAND ----------

df2.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Non parametric method: Kaplan–Meier estimator
# MAGIC 
# MAGIC -When we don't have any covariates in our data that time we use non parametric method.
# MAGIC 
# MAGIC -The Kaplan–Meier estimator is a non-parametric statistic used to estimate the survival function (probability of a person surviving) from lifetime data. In medical research, it is often used to measure the fraction of patients living for a certain amount of time after treatment. For example, Calculating the amount of time(year, month, day) certain patient lived after he/she was diagnosed with cancer or his treatment starts.
# MAGIC 
# MAGIC **s(t) = (No of subject at risk at start) -(No of subjects that died)/no. of subjects at start at risk at star**t 
# MAGIC 
# MAGIC -Nelson-Aalen hazard function:The survival functions are a great way to summarize and visualize the survival dataset. However, it is not the only way. If we are curious about the hazard function h(t) of a population, we, unfortunately, can’t transform the Kaplan Meier estimate. For that, we use the Nelson-Aalen hazard function
# MAGIC 
# MAGIC -clinical life tables: if we have larger time window then instead of checking each time point we create time window and then calculates values. It is similar as km analysis except we consider here time window
# MAGIC 
# MAGIC 
# MAGIC -**Assumtions**:
# MAGIC 
# MAGIC 1) Survival Probabilities are the same for all the samples who joined late in the study and those who have joined early. The Survival analysis which can affect is not assumed to change.
# MAGIC 
# MAGIC 2) Occurrence of Event are done at a specified time.
# MAGIC 
# MAGIC 3) Censoring of the study does not depend on the outcome. The Kaplan Meier method doesn’t depend on the outcome of interest.

# COMMAND ----------

# DBTITLE 1,Fitting a non-parametric model [Kaplan Meier Curve]
kmf = KaplanMeierFitter(alpha=0.05) # calculate a 95% confidence interval
kmf.fit(df2['tenure'], df2['target'])

#Conclusion: The output of this step tells us we fit the KM model using nearly 2.7 lac customer records of which 5587 didn't traded in quarter span(90 days). (The term right-censored tells us that the event of interest, i.e. trade, has not occurred within our observation window.) Using this model, we can now calculate the median survival time for any given customer.

# COMMAND ----------

# DBTITLE 1,Calculate Median Survival Time
median_ = kmf.median_survival_time_
median_

#We confirmed our past analysis that in frequent traders most of the customers are daily traders

# COMMAND ----------

# DBTITLE 1,Portion of Population Surviving at Point in Time
kmf.predict([7, 15, 30,90])

#Conclusion:We can say 13% didn't traded(survived in first week) and 87% people traded 

# COMMAND ----------

# DBTITLE 1,The Survival Rate over Time
# plot attributes
plt.figure(figsize=(25,15))
plt.title('Next trade duration', fontsize='xx-large')
# y-axis
plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.5))
plt.axhline(0.5, color='red', alpha=0.75, linestyle=':') # median line in red

# x-axis
plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(np.arange(0.0,90.0,7.0))
plt.axvline(7, color='gray', alpha=0.5, linestyle=':')  
plt.axvline(15, color='gray', alpha=0.5, linestyle=':')  
plt.axvline(30, color='gray', alpha=0.5, linestyle=':') 
plt.axvline(90, color='gray', alpha=0.5, linestyle=':') 

plt1 = kmf.plot_survival_function()
display(plt1)



# COMMAND ----------

#probablity that event occurs before time t      
#cd = 1 - survival
display(kmf.plot_cumulative_density())


# COMMAND ----------

#pending analysis each category wise

# COMMAND ----------

#Step 3: Examine How Survivorship Varies    ------- EDA part from notebook 2 skipped

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Semi parametric method: Cox proportional hazard model
# MAGIC 
# MAGIC This approach is referred to as a semi-parametric approach because while the hazard function is estimated non-parametrically, the functional form of the covariates is parametric.
# MAGIC 
# MAGIC **Cox proportional hazard model:**
# MAGIC 
# MAGIC works for both quantitative predictor (non-categorical) variables and categorical variables.
# MAGIC Why do we need it? In medical research, generally, we are considering more than one factor to diagnose a person’s health or survival time, i.e., we generally make use of their gender,  age, blood pressure, and blood sugar to find out if there is any significant difference between those in different groups. For example, if we are grouping our data based on a person’s age, then our goal will be to find out which age group has a higher survival chance. Is that the children’s group, adult’s group, or old person’s group? Now what we need to find is on what basis do we make the group? To find that we use Cox regression and find the coefficients of different parameters.
# MAGIC 
# MAGIC **Hazard and Hazard Ratio**
# MAGIC 
# MAGIC -Hazard is defined as the slope of the survival curve — a measure of how rapidly subjects are dying.
# MAGIC 
# MAGIC -The hazard ratio compares two treatments. If the hazard ratio is 2.0, then the rate of deaths in one treatment group is twice the rate in the other group.
# MAGIC 
# MAGIC -The HR greater than 1 indicates that as the value of ith covariate increases, the event hazard increases, and thus the duration of survival decreases.
# MAGIC 
# MAGIC HR = 1    No effect
# MAGIC 
# MAGIC HR < 1    Reduction in hazard
# MAGIC 
# MAGIC HR > 1    Increase in hazard
# MAGIC 
# MAGIC 
# MAGIC **Cox proportional hazards regression model assumptions includes:**
# MAGIC 
# MAGIC 1.Independence of survival times between distinct individuals in the sample 
# MAGIC 
# MAGIC 2.A multiplicative relationship between the predictors and the hazard
# MAGIC 
# MAGIC 3.A constant hazard ratio over time. This assumption implies that, the hazard curves for the groups should be proportional and cannot cross.

# COMMAND ----------

# DBTITLE 1,Encoding categorical variables
encoded_pd = pd.get_dummies(
    df2,
    columns=['occupation', 'Tier','new_incomedetails','gender_f','riskcategory_new'], 
    prefix=['occupation', 'Tier','new_incomedetails','gende_f','riskcategory_new'],
    drop_first=True
    )

encoded_pd.rename(columns={'occupation_Private Sector Service':'occupation_Private',
                           'occupation_Government Service':'occupation_Government'},inplace=True)

#encoded_pd.drop(['occupation_Forex Dealer'], axis=1, inplace=True)

encoded_pd.head()

# COMMAND ----------

#Doubt:We can see group size of strata variable is small i.e bottom groups have only size1. To avoid this instead of them using calculated labels 
encoded_pd.groupby('average_gap_last_four_trades_label_new').size()

# COMMAND ----------

# DBTITLE 1,Train the Model
#Instead of strata use average_gap_last_four_trades_label_new as baseline model, so it needs to drop 
#penalizer :Attach a penalty to the size of the coefficients during regression. This improves stability of the estimates and controls for high correlation between covariates. 
#l1 ratio default is 0 concept is same as penalizer
#h0(t) is the baseline hazard and represents the hazard when all of the predictors (or independent variables) X1, X2 , Xp are equal to zero. total 3 estimators used to calculate this breslow is one of them. Basically its cumulative hazard function at time t
#doubt: partial log-likelihood    ??????
#se(coef)?????
#what is use of last 4 columns  ???
#last 3 lines

cph = CoxPHFitter(alpha=0.05)     #95% confidence level
cph.fit(encoded_pd, 'tenure', 'target',strata='average_gap_last_four_trades_label_new')     # if strata used then its fine if coeeficients are not following assumption
cph.print_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Conclusion 
# MAGIC 
# MAGIC The model summary provides quite a bit of useful information but we should start with the p-scores associated with each feature. If any p-values greater 
# MAGIC than or equal to 0.05, the associated feature should be considered statistically insignificant (assuming a 95% threshold). and need to remove them
# MAGIC 
# MAGIC 
# MAGIC from above table exp(coef) are hazard ratio e.g occupation_Business = 1.02 means it increases prabablity of trade happening by 2%
# MAGIC These values are simple multiplier: Means if we want to see what is probablity of event happening for occupation business + tier II then = 1.02 * 0.99

# COMMAND ----------

plt.subplots(figsize=(10, 6))
display(cph.plot())

#Check which factor affects the most from the graph
#occupation forex dealer HR value is high


# COMMAND ----------

# MAGIC %md
# MAGIC #Validation of model:
# MAGIC 
# MAGIC The Cox PH model assumes that the hazard associated with a feature does not vary over time. A formal test of this assumption is provided through the check_assumptions() method on the model object. 
# MAGIC 
# MAGIC Even if the assumption was violated, more and more statisticians are making the case that models with light violations of the assumption may still be valid if the hazard ratios are taken as average hazards over the period for which the models were trained.
# MAGIC 
# MAGIC With larger dataset such as the one we are using, the check_assumptions() method is incredibly time-consuming. One trick that may help with reducing the time for this check is to randomly sample your data, retrain the model, and test the assumption off the newly trained version.

# COMMAND ----------

# giving problem to all features. In databricks also not used it
cph.check_assumptions(encoded_pd, p_value_threshold = 0.05)


# COMMAND ----------

#proportional_hazard_test: Alternate way to check assumption.

#If p less than 0.05 then we can not use this model  occupation_Business,occupation_no_data
#what is -log
#results = proportional_hazard_test(cph, encoded_pd, time_transform='rank')
#results.print_summary(decimals=3, model="untransformed variables")


#Removing as not understanding purpose

# COMMAND ----------

# DBTITLE 1,Access the Baseline Hazard
#In each row we can say bucket 1 has highest hazard value
baseline_chazard = cph.baseline_cumulative_hazard_
cph.baseline_cumulative_hazard_.head(30)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Notice how the hazard, i.e. rate of trade, increases with each passing day (recorded in the DataFrame's index). Notice too that each bucket/timeframe its own column of hazard data. It predicts that if customer is from week bucket then next day trading probablity is 0.7 and its increasing day by day as customer is becoming more frequent.

# COMMAND ----------

# DBTITLE 1,Extract the Coefficients
coefficients = cph.params_
coefficients

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Cox PH hazard rate formula:
# MAGIC 
# MAGIC **hazard rate = baseline hazard * partial hazard**

# COMMAND ----------

# DBTITLE 1,Retrieve Coefficients
# function to look up coefficient value
def get_coefficient(key, coefficients):
    if key in coefficients.keys():
        return coefficients[key]
    else:
        return 0

# COMMAND ----------

# DBTITLE 1,Apply Coefficients to Baseline to Predict Hazard
#feature values #for now taking dummy features as baseline in future it will get replaced by average_gsp_label
new_incomedetails=2  
Tier='Tier-III'       
average_gap_last_four_trades_label_new = 7
t = 9

# retreieve coefficients
new_incomedetails_coefficient = get_coefficient('new_incomedetails_{0}'.format(new_incomedetails), coefficients)
Tier_coefficient = get_coefficient('Tier_{0}'.format(Tier), coefficients) 

# calculate hazard
baseline_at_t = baseline_chazard.loc[t, average_gap_last_four_trades_label_new]
partial_hazard = np.exp(new_incomedetails_coefficient + Tier_coefficient)
hazard_rate = baseline_at_t * partial_hazard

# display results
print('Cumulative Hazard Rate @ Day {1}:\t{0:.4f}'.format(hazard_rate, t))
print('   Baseline Hazard @ Day {1}:\t\t{0:.4f}'.format(baseline_at_t, t))
print('   Partial Hazard:\t\t\t{0:.4f}'.format(partial_hazard))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Just to verify we are doing this correctly, let's compare our calculated result against the one returned by the lifelines model for this same subscription at the same point in time. We'll grab a subscriber from our input dataset that matches these criteria so that we don't have to juggle the encoding logic:

# COMMAND ----------

# DBTITLE 1,Use Model to Predict Hazard
# retreieve encoded value for our selected customer
X = encoded_pd[
  (df2['new_incomedetails']==new_incomedetails) &
  (df2['Tier']==Tier) &
  (df2['average_gap_last_four_trades_label_new']==average_gap_last_four_trades_label_new)
  ].head(1)

# predict cumulative hazard
cph.predict_cumulative_hazard(X, times=[t])

# COMMAND ----------

# MAGIC %md
# MAGIC The model produces a different result from the one we calculated!!! It is because of partial hazard

# COMMAND ----------

# DBTITLE 1,Get the Partial Hazard for the Baseline (Reference) Member
#feature values  
new_incomedetails=2  
Tier='Tier-III'       
average_gap_last_four_trades_label_new = 7

# retrieve a baseline subscription
X = encoded_pd[
  (df2['new_incomedetails']==new_incomedetails) &
  (df2['Tier']==Tier) &
  (df2['average_gap_last_four_trades_label_new']==average_gap_last_four_trades_label_new)
  ].head(1)

# get the partial hazard (factor) for the baseline member
hidden_partial_hazard = cph.predict_partial_hazard(X).iloc[0]
print(hidden_partial_hazard)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC  By applying partial hazard to the baseline, we can correct the issue, allowing us to use the formula above without additional modification:
# MAGIC 
# MAGIC NOTE The hidden partial hazard is a rate, not a coefficient; it has already been transformed into a factor which can be applied directly to the baseline hazard.

# COMMAND ----------

# DBTITLE 1,Adjust the Baseline Hazard for the Hidden Partial Hazard
# if baseline factor is not present, set it to 1 so that math works in next step
if np.isnan(hidden_partial_hazard): hidden_partial_hazard=1

# adjust the baseline by the baseline factor
baseline_chazard_adj= baseline_chazard * hidden_partial_hazard
baseline_chazard_adj.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC With our baseline hazard adjusted, let's now attempt compare our results to those returned by the model:

# COMMAND ----------

# DBTITLE 1,Calculate Hazard
#feature values
new_incomedetails=2  
Tier='Tier-III'       
average_gap_last_four_trades_label_new = 7
t = 9

# calculate hazard
baseline_at_t = baseline_chazard_adj.loc[t, average_gap_last_four_trades_label_new]
partial_hazard = np.exp(new_incomedetails_coefficient + Tier_coefficient)
hazard_rate = baseline_at_t * partial_hazard

# display results
print('Cumulative Hazard Rate @ Day {1}:\t{0:.4f}'.format(hazard_rate, t))
print('   Baseline Hazard @ Day {1}:\t\t{0:.4f}'.format(baseline_at_t, t))
print('   Partial Hazard:\t\t\t{0:.4f}'.format(partial_hazard))


# COMMAND ----------

# DBTITLE 1,Use Model to Predict Hazard
#feature values 
new_incomedetails=2  
Tier='Tier-III'       
average_gap_last_four_trades_label_new = 7
t = 9

# baseline member
X = encoded_pd[
  (df2['new_incomedetails']==new_incomedetails) &
  (df2['Tier']==Tier) &
  (df2['average_gap_last_four_trades_label_new']==average_gap_last_four_trades_label_new)
  ].head(1)

# model prediction
cph.predict_cumulative_hazard(X, times=[t])

# COMMAND ----------

# MAGIC %md
# MAGIC #Extract the Data Required for Survival Ratio Calculations
# MAGIC 
# MAGIC The baseline hazard formula is the most widely documented formula surrounding Cox PH models. Still, baseline hazard is not an intuitive value for most analysts. Instead, most analysts tend to think about survival over time in terms of the ratio of a given population that survives to a given point in time. This value is known as the survival ratio:
# MAGIC 
# MAGIC As with the baseline hazard rate, the survival ratio can be extracted and must be adjusted for the hidden partial hazard:

# COMMAND ----------

# DBTITLE 1,Extract Baseline Survival Data
# retrieve baseline survival ratio 
baseline_survival = cph.baseline_survival_
baseline_survival.head(10)

# COMMAND ----------

# DBTITLE 1,Retrieve Hidden Partial Hazard (Same as Above)
#feature values

new_incomedetails=2  
Tier='Tier-III'       
average_gap_last_four_trades_label_new = 7

# retrieve a baseline subscription
X = encoded_pd[
  (df2['new_incomedetails']==new_incomedetails) &
  (df2['Tier']==Tier) &
  (df2['average_gap_last_four_trades_label_new']==average_gap_last_four_trades_label_new)
  ].head(1)

# get the partial hazard (factor) for the baseline member
hidden_partial_hazard = cph.predict_partial_hazard(X).iloc[0]

# if baseline factor is not present, set it to 1 so that math works in next step
if np.isnan(hidden_partial_hazard): hidden_partial_hazard=1

# COMMAND ----------

# DBTITLE 1,Adjust Baseline Survival by Hidden Partial Hazard
# adjust the baseline by the baseline factor
baseline_survival_adj= np.power(baseline_survival, hidden_partial_hazard)
baseline_survival_adj.head(10)

# COMMAND ----------

# DBTITLE 1,Calculate Survival Ratio
#feature values
new_incomedetails=2  
Tier='Tier-III'       
average_gap_last_four_trades_label_new = 7
t = 9

# retreive feature coefficients
new_incomedetails_coefficient = get_coefficient('new_incomedetails_{0}'.format(new_incomedetails), coefficients) 
Tier_coefficient = get_coefficient('Tier_{0}'.format(Tier), coefficients) 

# calculate hazard ratio
baseline_at_t = baseline_survival_adj.loc[t, average_gap_last_four_trades_label_new]
partial_hazard = np.exp(new_incomedetails_coefficient + Tier_coefficient)
survival_rate = np.power(baseline_at_t, partial_hazard)

print('Calculated:\t{0:.4f}'.format(survival_rate))

# COMMAND ----------

# DBTITLE 1,Retrieve Survival Rate
#feature values
new_incomedetails=2  
Tier='Tier-III'       
average_gap_last_four_trades_label_new = 7
t = 9

X = encoded_pd[
  (df2['new_incomedetails']==new_incomedetails) &
  (df2['Tier']==Tier) &
  (df2['average_gap_last_four_trades_label_new']==average_gap_last_four_trades_label_new)
  ].head(1)

# model prediction
cph.predict_survival_function(X, times=[t])


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Again, pre-applying the partial hazard to our baseline survival ratio allows us to generate a survival ratio prediction that matches what the model returns.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Parametric method:  [Accelerated Failure Time Model (AFT)]

# COMMAND ----------

# MAGIC %md
# MAGIC -The canonical example of Accelerated Failure Time models, shared by Kleinbaum & Klein in Survival Analysis: A Self-Learning Text, is the lifespan of dogs. It is commonly accepted that dogs age 7x faster than humans. They go through the same lifestages that we do, just faster.
# MAGIC 
# MAGIC -In contrast to Kaplan-Meier and Cox Proportional Hazards, Accelerated Failure Time is a parametric model. This means that the outcome variable is assumed to follow a specified distribution. Parametric models are typically less 'flexible' than non-parametric and semi-parametric models but can be a good choice when you're comfortable with specifying the distribution of the outcome variable.
# MAGIC 
# MAGIC 
# MAGIC **The Accelerated Failure Time Model Equation**
# MAGIC 
# MAGIC 1.Using the Accelerated Failure Time equation below, if we were to define group A as humans and group B as dogs, then the acceleration factor would be 7. Similarly, if we define group A as dogs and group B as humans, then the acceleration factor would be 1/7.
# MAGIC 
# MAGIC 2.The specification for lambda, which represents the accelerated failure rate, is intentionally generalized here. In practice, the survival function for the accelerated failure rate includes one or more parameters. For example, the specification when using log-logistic accelerated failure time is: 1/(1+lambda x t ^ p).
# MAGIC 
# MAGIC 3.The full specification of the accelerated failure rate is most relevant when using log-log plots to verify whether the accelerated failure time assumptions have been violated. 
# MAGIC 
# MAGIC Survival function(Group A) = Survival function (Group B) * Accelerated Failure Rate (lambda)
# MAGIC 
# MAGIC **Assumptions:**
# MAGIC AFT model assumes that the effect of a covariate is to accelerate or decelerate the life course of a event by some constant.

# COMMAND ----------

df2.head(10)

# COMMAND ----------

encode_cols = ['occupation','Tier','new_incomedetails','gender_f','riskcategory_new','average_gap_last_four_trades_label_new']
 
encoded_pd1 = pd.get_dummies(df2,
               columns=encode_cols,
               prefix=encode_cols,
               drop_first=True)
 
encoded_pd1.head()

# COMMAND ----------

# Instantiate each fitter
wb = WeibullFitter()
ex = ExponentialFitter()
log = LogNormalFitter()
loglogis = LogLogisticFitter()

# Fit to data
for model in [wb, ex, log, loglogis]:
    model.fit(durations = encoded_pd1["tenure"],
              event_observed = encoded_pd1["target"])
    # Print AIC
    print("The AIC value for", model.__class__.__name__, "is",  model.AIC_)

# COMMAND ----------

#Fit the LogLogisticFitter fitter and print summary (AIC gives which model has less error rate. In our case LogLogisticFitter has minimum AIC)

logistic_aft = LogLogisticFitter()
logistic_aft.fit(encoded_pd1['tenure'], encoded_pd1['target'])

logistic_aft.print_summary(3)

# COMMAND ----------

# MAGIC %md
# MAGIC #Interpretation of the coefficients
# MAGIC 
# MAGIC A unit increase in  means the average/median survival time changes by a factor of .
# MAGIC Suppose  was positive, then the factor exp(b) is greater than 1, which will decelerate the event time since we divide time by the factor ⇿ increase mean/median survival. Hence, it will be a protective effect.
# MAGIC Likewise, a negative  will hasten the event time ⇿ reduce the mean/median survival time.

# COMMAND ----------

plt5 = logistic_aft.plot()
display(plt5)

#If line comes straight then we can use this model for data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #References:
# MAGIC 
# MAGIC https://databricks.com/notebooks/survival_analysis/survival_analysis_01_data_prep.html
# MAGIC 
# MAGIC https://databricks.com/notebooks/survival_analysis/survival_analysis_02_exploratory_analysis.html
# MAGIC 
# MAGIC https://databricks.com/notebooks/survival_analysis/survival_analysis_03_modeling_hazards.html
# MAGIC 
# MAGIC https://databricks.com/notebooks/survival_analysis/survival_analysis_04_operationalization.html
# MAGIC 
# MAGIC https://databricks.com/notebooks/telco-accel/05_customer_lifetime_value.html
# MAGIC 
# MAGIC https://databricks.com/blog/2020/08/07/on-demand-virtual-workshop-predicting-churn-to-improve-customer-retention.html
# MAGIC 
# MAGIC https://databricks.com/notebooks/telco-accel/04_accelerated_failure_time.html   (AFT)
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.AFTSurvivalRegression.html
# MAGIC 
# MAGIC https://blog.damavis.com/en/application-of-survival-analysis-to-price-changes/
# MAGIC 
# MAGIC https://www.youtube.com/watch?v=GpIk1NhZiVU 
# MAGIC 
# MAGIC https://towardsdatascience.com/survival-analysis-part-a-70213df21c2e(Censorship)
# MAGIC 
# MAGIC https://www.kdnuggets.com/2020/07/complete-guide-survival-analysis-python-part1.html (definition)
# MAGIC 
# MAGIC https://pypi.org/project/lifelines/ 
# MAGIC 
# MAGIC https://towardsdatascience.com/survival-analysis-to-understand-customer-retention-e3724f3f7ea2 (Concordance Index)
# MAGIC 
# MAGIC https://www.youtube.com/watch?v=fTX8GghbBPc
# MAGIC 
# MAGIC https://lifelines.readthedocs.io/en/latest/fitters/univariate/LogLogisticFitter.html
# MAGIC 
# MAGIC 
# MAGIC If any error, refer site:
# MAGIC 
# MAGIC https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-modelMatrix%20is%20singular
# MAGIC 
# MAGIC Other:
# MAGIC 
# MAGIC https://medium.com/the-researchers-guide/survival-analysis-in-python-km-estimate-cox-ph-and-aft-model-5533843c5d5d
# MAGIC 
# MAGIC https://onezero.blog/survival-analysis-in-python-km-estimate-cox-ph-and-aft-model/
