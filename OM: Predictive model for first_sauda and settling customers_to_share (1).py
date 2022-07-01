# Databricks notebook source
 spark.catalog.clearCache()

# COMMAND ----------

# DBTITLE 1,Importing packages
from pyspark.sql.functions import col
import pandas as pd
from databricks import automl
import mlflow
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report,roc_curve,confusion_matrix,f1_score
from mlflow.tracking import MlflowClient
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn import set_config
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pyspark.sql.types import ArrayType,StringType,IntegerType,BooleanType
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

# COMMAND ----------

# DBTITLE 1,train_end_date widget
# MAGIC %sql
# MAGIC CREATE WIDGET DROPDOWN train_end_date DEFAULT "2022-02-01" CHOICES 
# MAGIC select distinct sauda_date as end_date from ora_dormancy.user_daily_trade_summary where sauda_date >= '2022-02-01' order by end_date

# COMMAND ----------

# DBTITLE 1,train_start_date widget
# MAGIC %sql
# MAGIC 
# MAGIC CREATE WIDGET DROPDOWN train_start_date DEFAULT "2021-12-01" CHOICES 
# MAGIC select distinct sauda_date as start_date from ora_dormancy.user_daily_trade_summary where sauda_date >= '2021-12-01' order by start_date

# COMMAND ----------

# DBTITLE 1,test_end_date widget
# MAGIC %sql
# MAGIC CREATE WIDGET DROPDOWN test_end_date DEFAULT "2022-02-01" CHOICES 
# MAGIC select distinct sauda_date as end_date from ora_dormancy.user_daily_trade_summary where sauda_date >= '2022-02-01' order by end_date

# COMMAND ----------

# DBTITLE 1,test_start_date widget
# MAGIC %sql
# MAGIC CREATE WIDGET DROPDOWN test_start_date DEFAULT "2021-12-01" CHOICES 
# MAGIC select distinct sauda_date as start_date from ora_dormancy.user_daily_trade_summary where sauda_date >= '2021-12-01' order by start_date

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Common train data preparation

# COMMAND ----------

# DBTITLE 1,Train: ora_dormancy.user_daily_trade_summary
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW training_data as
# MAGIC with training_intermediate as (
# MAGIC   select
# MAGIC     *,
# MAGIC     date_add(sauda_date, 30) as cutoff_sauda_date
# MAGIC   from
# MAGIC     ora_dormancy.user_daily_trade_summary
# MAGIC   WHERE
# MAGIC     cast(sauda_date as date) >= cast('$train_start_date' as date)
# MAGIC     AND cast(sauda_date as date) <= cast('$train_end_date' as date)
# MAGIC ),
# MAGIC training_target as(
# MAGIC   select
# MAGIC     *,
# MAGIC     max(cumulative_trade_days) OVER (
# MAGIC       PARTITION BY party_code
# MAGIC       ORDER BY
# MAGIC         sauda_date BETWEEN sauda_date
# MAGIC         AND cutoff_sauda_date
# MAGIC     ) AS target_trade_days
# MAGIC   from
# MAGIC     training_intermediate
# MAGIC )
# MAGIC select
# MAGIC   *,
# MAGIC   CASE
# MAGIC     WHEN target_trade_days = 1 THEN 0
# MAGIC     WHEN target_trade_days IN (2, 3) THEN 1
# MAGIC     WHEN target_trade_days > 3  THEN 2
# MAGIC   END AS settle_target,
# MAGIC   CASE
# MAGIC     WHEN target_trade_days > 3 THEN 1
# MAGIC     WHEN target_trade_days IN (2, 3)
# MAGIC     and cumulative_trade_days > 1 THEN 0
# MAGIC   END AS frequent_target
# MAGIC from
# MAGIC   training_target

# COMMAND ----------

# DBTITLE 1,Train: amx.order_detail
# MAGIC %sql
# MAGIC 
# MAGIC --underlying_assets: currently present assets with user. Get count as a feature 
# MAGIC -- rejected : orders got rejected. get count as a feature 
# MAGIC -- day validity : time for which order will be valid 
# MAGIC -- immediate or cancel: execute immediate
# MAGIC -- good till days: Very rarely people use it
# MAGIC -- good till cancel: Very rarely people use it
# MAGIC --end of session What is status of orders on end of session
# MAGIC --delivery cash: cash invested in delivery
# MAGIC --margin intra day: margin/loan for intraday
# MAGIC -- margin delivery: margin/loan for delivery
# MAGIC --Normal_fno : Its regular fno only
# MAGIC --Bracket_order : type of order get more information on google
# MAGIC --Arbitrage_order: type of order get more information on google
# MAGIC --Cover_order:type of order get more information on google
# MAGIC --Market_clearing: based on market value
# MAGIC --Limit_clearing : limit amount set to buy or sell
# MAGIC --Stop_loss_clearing : stop loss set to buy
# MAGIC -- buy: no. of buy
# MAGIC -- sell: no. of sell
# MAGIC -- no group: If no speacial equity then xx
# MAGIC -- special equity: If there is special equity then value upated here eg BT: bulk trading
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW daily_order_log as
# MAGIC with ord_log as (
# MAGIC   select
# MAGIC     distinct straccountid as party_id,
# MAGIC     strguiorgordid as trade_id,
# MAGIC     cast(dt as Date) as sauda_date2,
# MAGIC     strordduration as order_validity,
# MAGIC     strproduct as product_type,
# MAGIC     strordertype as exec_price,
# MAGIC     strtranstype as buy_or_sell,
# MAGIC     strstrategycode as strategy,
# MAGIC     strinsttype as instrument,
# MAGIC     strgroup as grouped_as,
# MAGIC     strsymbolname as underlying_assets,
# MAGIC     strrejectionby as rejected_by
# MAGIC   from
# MAGIC     amx.order_detail
# MAGIC   where
# MAGIC   cast(dt as Date)>= cast('$train_start_date' as date)
# MAGIC ) 
# MAGIC select
# MAGIC   party_id,
# MAGIC   sauda_date2,
# MAGIC   collect_set(underlying_assets) as underlying_assets,
# MAGIC   collect_list(rejected_by) as rejected_by,
# MAGIC   sum(case
# MAGIC     when order_validity = 'DAY' then 1
# MAGIC     else 0 end)
# MAGIC   as Day_validity,
# MAGIC   sum(case
# MAGIC     when order_validity = 'IOC' then 1
# MAGIC     else 0 end)
# MAGIC   as Immediate_or_Cancel,
# MAGIC   sum(case
# MAGIC     when order_validity = 'GTD' then 1
# MAGIC     else 0 end)
# MAGIC   as Good_till_days,
# MAGIC   sum(case
# MAGIC     when order_validity = 'GTC' then 1
# MAGIC     else 0 end)
# MAGIC   as Good_till_cancel,
# MAGIC   sum(case
# MAGIC     when order_validity = 'EOS' then 1
# MAGIC     else 0 end)
# MAGIC   as End_of_Session,
# MAGIC   sum(case
# MAGIC     when product_type = 'CNC' then 1
# MAGIC     else 0 end)
# MAGIC   as Delivery_cash,
# MAGIC   sum(case
# MAGIC     when product_type = 'MIS' then 1
# MAGIC     else 0 end)
# MAGIC   as Margin_intraday,
# MAGIC   sum(case
# MAGIC     when product_type = 'MARGIN' then 1
# MAGIC     else 0 end)
# MAGIC   as Margin_delivery,
# MAGIC   sum(case
# MAGIC     when product_type = 'NRML' then 1
# MAGIC     else 0 end)
# MAGIC   as Normal_fno,
# MAGIC   sum(case
# MAGIC     when product_type = 'BO' then 1
# MAGIC     else 0 end)
# MAGIC   as Bracket_order,
# MAGIC   sum(case
# MAGIC     when product_type = 'ARB' then 1
# MAGIC     else 0 end)
# MAGIC   as Arbitrage_order,
# MAGIC   sum(case
# MAGIC     when product_type = 'CO' then 1
# MAGIC     else 0 end)
# MAGIC   as Cover_order,
# MAGIC   sum(case
# MAGIC     when exec_price = 'MKT' then 1
# MAGIC     else 0 end)
# MAGIC   as Market_clearing,
# MAGIC   sum(case
# MAGIC     when exec_price = 'L' then 1
# MAGIC     else 0 end)
# MAGIC   as Limit_clearing,
# MAGIC   sum(case
# MAGIC     when exec_price = 'SL' then 1
# MAGIC     else 0 end)
# MAGIC   as Stop_loss_clearing,
# MAGIC   sum(case
# MAGIC     when buy_or_sell = 'B' then 1
# MAGIC     else 0 end)
# MAGIC   as Buy,
# MAGIC   sum(case
# MAGIC     when buy_or_sell = 'S' then 1
# MAGIC     else 0 end)
# MAGIC   as Sell,
# MAGIC   sum(case
# MAGIC     when grouped_as = 'XX' then 1
# MAGIC     else 0 end)
# MAGIC   as No_group,
# MAGIC   sum(case
# MAGIC     when grouped_as != 'EQ'
# MAGIC     and grouped_as != 'XX' then 1
# MAGIC     else 0 end)
# MAGIC   as Special_eq
# MAGIC from
# MAGIC   ord_log
# MAGIC group by
# MAGIC 1,2

# COMMAND ----------

# DBTITLE 1,Common train data processing for both models
df_sn = spark.read.table('online_engine.client_kyc')
df_sn.createOrReplaceTempView("SN_ClientKYC")
df_oc = spark.read.table('online_engine.order_count')
df_oc.createOrReplaceTempView("order_count")

df = sqlContext.sql(f""" 
select * from training_data a 
left join 
(select Party_code as party_code2,sauda_date as sauda_date1,sum(Brokerage) as Brokerage ,sum(T_O) as T_O from online_engine.order_count group by party_code2,sauda_date1) b 
on a.party_code = b.party_code2 and a.sauda_date = b.sauda_date1
left join 
(select * from daily_order_log)c
on b.party_code2 = c.party_id and b.sauda_date1=c.sauda_date2
left join 
(select distinct party_code as party_code1, gender,riskcategory,incomedetails,occupation,City,round(((datediff((select current_date()),Birthdate))/365),2) as age 
 from SN_ClientKYC)d on c.party_id = d.party_code1 
""")

def list_remove_space(original_list):    
    if original_list != None:
        return [i for i in original_list if i]

def udf_count(new_list):
    if new_list != None:
        return len(new_list)
    
udf_remove_space = udf(lambda x:list_remove_space(x),ArrayType(StringType()))
udf_count_final = udf(lambda x:udf_count(x),IntegerType())
df = df.withColumn("rejected_by_new",udf_remove_space('rejected_by'))
df = df.withColumn("rejected_by_cnt",udf_count_final('rejected_by_new'))
df = df.withColumn("underlying_assets_new",udf_remove_space('underlying_assets'))
df = df.withColumn("underlying_assets_cnt",udf_count_final('underlying_assets_new'))

df=df.drop('average_gap_last_four_trades_weeks','average_gap_last_four_trades_label','rejected_by','underlying_assets','rejected_by_new','underlying_assets_new')

df= df.na.fill({"has_mis_trade":0,"has_cnc_trade":0,"has_fut_idx_trade":0,"has_fut_stk_trade":0,"has_opt_idx_trade":0,"has_opt_stk_trade":0,"has_curr_trade":0,"has_comm_trade":0})

df = df.select(
    (df.has_mis_trade.cast(BooleanType()))
    .alias('has_mis_trade'),
    (df.has_cnc_trade.cast(BooleanType()))
    .alias('has_cnc_trade'),
    (df.has_fut_idx_trade.cast(BooleanType()))
    .alias('has_fut_idx_trade'),
    (df.has_fut_stk_trade.cast(BooleanType()))
    .alias('has_fut_stk_trade'),
    (df.has_opt_idx_trade.cast(BooleanType()))
    .alias('has_opt_idx_trade'),
    (df.has_opt_stk_trade.cast(BooleanType()))
    .alias('has_opt_stk_trade'),
    (df.has_curr_trade.cast(BooleanType()))
    .alias('has_curr_trade'),
    (df.has_comm_trade.cast(BooleanType()))
    .alias('has_comm_trade'),df.settle_target,df.frequent_target,
    df.gender,
    df.riskcategory,
    df.incomedetails,
    df.occupation,
    df.City,
    df.party_code,
    df.party_code1,
    df.party_code2,
    df.party_id,
    df.sauda_date,
    df.sauda_date1,
    df.sauda_date2,
    df.cutoff_sauda_date,
    df.cumulative_trade_days,
    df.target_trade_days,
    df.Brokerage,
    df.T_O,
    df.Day_validity,
    df.Immediate_or_Cancel,
    df.Good_till_days,
    df.Good_till_cancel,
    df.End_of_Session,
    df.Delivery_cash,
    df.Margin_intraday,
    df.Margin_delivery,
    df.Normal_fno,
    df.Bracket_order,
    df.Arbitrage_order,
    df.Cover_order,
    df.Market_clearing,
    df.Limit_clearing,
    df.Stop_loss_clearing,
    df.Buy,
    df.Sell,
    df.No_group,
    df.Special_eq,
    df.rejected_by_cnt,
    df.underlying_assets_cnt,
    df.age
)


# COMMAND ----------

# MAGIC %md
# MAGIC #Common test data preparation

# COMMAND ----------

# DBTITLE 1,Test: ora_dormancy.user_daily_trade_summary
# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW testing_data as
# MAGIC with testing_intermediate as (
# MAGIC   select
# MAGIC     *,
# MAGIC     date_add(sauda_date, 30) as cutoff_sauda_date
# MAGIC   from
# MAGIC     ora_dormancy.user_daily_trade_summary
# MAGIC   WHERE
# MAGIC     cast(sauda_date as date) >= cast('$test_start_date' as date)
# MAGIC     AND cast(sauda_date as date) <= cast('$test_end_date' as date)
# MAGIC ),
# MAGIC testing_target as(
# MAGIC   select
# MAGIC     *,
# MAGIC     max(cumulative_trade_days) OVER (
# MAGIC       PARTITION BY party_code
# MAGIC       ORDER BY
# MAGIC         sauda_date BETWEEN sauda_date
# MAGIC         AND cutoff_sauda_date
# MAGIC     ) AS target_trade_days
# MAGIC   from
# MAGIC     testing_intermediate
# MAGIC )
# MAGIC select
# MAGIC   *,
# MAGIC   CASE
# MAGIC     WHEN target_trade_days = 1 THEN 0
# MAGIC     WHEN target_trade_days IN (2, 3) THEN 1
# MAGIC     WHEN target_trade_days > 3  THEN 2
# MAGIC   END AS settle_target,
# MAGIC   CASE
# MAGIC     WHEN target_trade_days > 3 THEN 1
# MAGIC     WHEN target_trade_days IN (2, 3)
# MAGIC     and cumulative_trade_days > 1 THEN 0
# MAGIC   END AS frequent_target
# MAGIC from
# MAGIC   testing_target

# COMMAND ----------

# DBTITLE 1,Test: amx.order_detail
# MAGIC %sql
# MAGIC -- TODO Materialise this table as intermediate
# MAGIC CREATE OR REPLACE TEMPORARY VIEW daily_order_log1 as
# MAGIC with ord_log as (
# MAGIC   select
# MAGIC     distinct straccountid as party_id,
# MAGIC     strguiorgordid as trade_id,
# MAGIC     cast(dt as Date) as sauda_date2,
# MAGIC     strordduration as order_validity,
# MAGIC     strproduct as product_type,
# MAGIC     strordertype as exec_price,
# MAGIC     strtranstype as buy_or_sell,
# MAGIC     strstrategycode as strategy,
# MAGIC     strinsttype as instrument,
# MAGIC     strgroup as grouped_as,
# MAGIC     strsymbolname as underlying_assets,
# MAGIC     strrejectionby as rejected_by
# MAGIC   from
# MAGIC     amx.order_detail
# MAGIC   where
# MAGIC   cast(dt as Date)>= cast('$test_start_date' as date)
# MAGIC ) 
# MAGIC select
# MAGIC   party_id,
# MAGIC   sauda_date2,
# MAGIC   collect_set(underlying_assets) as underlying_assets,
# MAGIC   collect_list(rejected_by) as rejected_by,
# MAGIC   sum(case
# MAGIC     when order_validity = 'DAY' then 1
# MAGIC     else 0 end)
# MAGIC   as Day_validity,
# MAGIC   sum(case
# MAGIC     when order_validity = 'IOC' then 1
# MAGIC     else 0 end)
# MAGIC   as Immediate_or_Cancel,
# MAGIC   sum(case
# MAGIC     when order_validity = 'GTD' then 1
# MAGIC     else 0 end)
# MAGIC   as Good_till_days,
# MAGIC   sum(case
# MAGIC     when order_validity = 'GTC' then 1
# MAGIC     else 0 end)
# MAGIC   as Good_till_cancel,
# MAGIC   sum(case
# MAGIC     when order_validity = 'EOS' then 1
# MAGIC     else 0 end)
# MAGIC   as End_of_Session,
# MAGIC   sum(case
# MAGIC     when product_type = 'CNC' then 1
# MAGIC     else 0 end)
# MAGIC   as Delivery_cash,
# MAGIC   sum(case
# MAGIC     when product_type = 'MIS' then 1
# MAGIC     else 0 end)
# MAGIC   as Margin_intraday,
# MAGIC   sum(case
# MAGIC     when product_type = 'MARGIN' then 1
# MAGIC     else 0 end)
# MAGIC   as Margin_delivery,
# MAGIC   sum(case
# MAGIC     when product_type = 'NRML' then 1
# MAGIC     else 0 end)
# MAGIC   as Normal_fno,
# MAGIC   sum(case
# MAGIC     when product_type = 'BO' then 1
# MAGIC     else 0 end)
# MAGIC   as Bracket_order,
# MAGIC   sum(case
# MAGIC     when product_type = 'ARB' then 1
# MAGIC     else 0 end)
# MAGIC   as Arbitrage_order,
# MAGIC   sum(case
# MAGIC     when product_type = 'CO' then 1
# MAGIC     else 0 end)
# MAGIC   as Cover_order,
# MAGIC   sum(case
# MAGIC     when exec_price = 'MKT' then 1
# MAGIC     else 0 end)
# MAGIC   as Market_clearing,
# MAGIC   sum(case
# MAGIC     when exec_price = 'L' then 1
# MAGIC     else 0 end)
# MAGIC   as Limit_clearing,
# MAGIC   sum(case
# MAGIC     when exec_price = 'SL' then 1
# MAGIC     else 0 end)
# MAGIC   as Stop_loss_clearing,
# MAGIC   sum(case
# MAGIC     when buy_or_sell = 'B' then 1
# MAGIC     else 0 end)
# MAGIC   as Buy,
# MAGIC   sum(case
# MAGIC     when buy_or_sell = 'S' then 1
# MAGIC     else 0 end)
# MAGIC   as Sell,
# MAGIC   sum(case
# MAGIC     when grouped_as = 'XX' then 1
# MAGIC     else 0 end)
# MAGIC   as No_group,
# MAGIC   sum(case
# MAGIC     when grouped_as != 'EQ'
# MAGIC     and grouped_as != 'XX' then 1
# MAGIC     else 0 end)
# MAGIC   as Special_eq
# MAGIC from
# MAGIC   ord_log
# MAGIC group by
# MAGIC 1,2

# COMMAND ----------

# DBTITLE 1,Common test data processing for both models
df1 = sqlContext.sql(f""" 
select * from testing_data a 
left join 
(select Party_code as party_code2,sauda_date as sauda_date1,sum(Brokerage) as Brokerage ,sum(T_O) as T_O from online_engine.order_count group by party_code2,sauda_date1) b 
on a.party_code = b.party_code2 and a.sauda_date = b.sauda_date1
left join 
(select * from daily_order_log1)c
on b.party_code2 = c.party_id and b.sauda_date1=c.sauda_date2
left join 
(select distinct party_code as party_code1, gender,riskcategory,incomedetails,occupation,City,round(((datediff((select current_date()),Birthdate))/365),2) as age 
from SN_ClientKYC)d on c.party_id = d.party_code1 

""")

def list_remove_space(original_list):
      if original_list != None:
        return [i for i in original_list if i]

def udf_count(new_list):
    if new_list != None:
        return len(new_list)
    
udf_remove_space = udf(lambda x:list_remove_space(x),ArrayType(StringType()))
udf_count_final = udf(lambda x:udf_count(x),IntegerType())
df1 = df1.withColumn("rejected_by_new",udf_remove_space('rejected_by'))
df1 = df1.withColumn("rejected_by_cnt",udf_count_final('rejected_by_new'))
df1 = df1.withColumn("underlying_assets_new",udf_remove_space('underlying_assets'))
df1 = df1.withColumn("underlying_assets_cnt",udf_count_final('underlying_assets_new'))

df1=df1.drop('average_gap_last_four_trades_weeks','average_gap_last_four_trades_label','rejected_by','underlying_assets','rejected_by_new','underlying_assets_new')

df1= df1.na.fill({"has_mis_trade":0,"has_cnc_trade":0,"has_fut_idx_trade":0,"has_fut_stk_trade":0,"has_opt_idx_trade":0,"has_opt_stk_trade":0,"has_curr_trade":0,"has_comm_trade":0})

df1 = df1.select(
    (df1.has_mis_trade.cast(BooleanType()))
    .alias('has_mis_trade'),
    (df1.has_cnc_trade.cast(BooleanType()))
    .alias('has_cnc_trade'),
    (df1.has_fut_idx_trade.cast(BooleanType()))
    .alias('has_fut_idx_trade'),
    (df1.has_fut_stk_trade.cast(BooleanType()))
    .alias('has_fut_stk_trade'),
    (df1.has_opt_idx_trade.cast(BooleanType()))
    .alias('has_opt_idx_trade'),
    (df1.has_opt_stk_trade.cast(BooleanType()))
    .alias('has_opt_stk_trade'),
    (df1.has_curr_trade.cast(BooleanType()))
    .alias('has_curr_trade'),
    (df1.has_comm_trade.cast(BooleanType()))
    .alias('has_comm_trade'),df1.settle_target,df1.frequent_target,
    df1.gender,
    df1.riskcategory,
    df1.incomedetails,
    df1.occupation,
    df1.City,
    df1.party_code,
    df1.party_code1,
    df1.party_code2,
    df1.party_id,
    df1.sauda_date,
    df1.sauda_date1,
    df1.sauda_date2,
    df1.cutoff_sauda_date,
    df1.cumulative_trade_days,
    df1.target_trade_days,
    df1.Brokerage,
    df1.T_O,
    df1.Day_validity,
    df1.Immediate_or_Cancel,
    df1.Good_till_days,
    df1.Good_till_cancel,
    df1.End_of_Session,
    df1.Delivery_cash,
    df1.Margin_intraday,
    df1.Margin_delivery,
    df1.Normal_fno,
    df1.Bracket_order,
    df1.Arbitrage_order,
    df1.Cover_order,
    df1.Market_clearing,
    df1.Limit_clearing,
    df1.Stop_loss_clearing,
    df1.Buy,
    df1.Sell,
    df1.No_group,
    df1.Special_eq,
    df1.rejected_by_cnt,
    df1.underlying_assets_cnt,
    df1.age
)



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Model1 : Using first trade customers data to predict class of customers i.e first_trade,settling,frequent

# COMMAND ----------

# DBTITLE 1,Feature engineering for settle target train data
df_settle1 = df.filter((df['cumulative_trade_days'] == 1))
df_settle1 = df_settle1.filter((df_settle1['sauda_date'] <= '2022-02-03'))
df_settle2 = df_settle1.drop('frequent_target')
df_settle1.unpersist()
df_settle=df_settle2.toPandas()

tier_info1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/Indian_Cities_by_Tier_I_and_II.csv')
tier_info=tier_info1.toPandas()
tier_info.dropna(subset=['Tier'], inplace=True)
tier_mapper = tier_info[['City', 'Tier']].set_index('City').to_dict()['Tier']    
def tier_mapping(city):
    try: return tier_mapper[city.capitalize()]
    except: return 'Tier-III'
df_settle['Tier'] = df_settle['City'].map(tier_mapping)
df_settle["gender"].replace({"f": "F"}, inplace=True)
df_settle.drop(['sauda_date','sauda_date1','sauda_date2','party_code','party_code1','party_code2','party_id','cutoff_sauda_date','cumulative_trade_days',
               'target_trade_days','City'], axis=1, inplace=True)

print(df_settle.shape)
print(df_settle['settle_target'].value_counts())  
display(df_settle)

# COMMAND ----------

# DBTITLE 1,Distribution of features
for col in df_settle.columns:
    print(df_settle[col].value_counts())
    

# COMMAND ----------

# DBTITLE 1,Feature engineering for settle target test data
df_settle_1 = df1.filter((df1['cumulative_trade_days'] == 1))
df_settle_1 = df_settle_1.filter((df_settle_1['sauda_date'] <= '2022-04-15'))

df_settle_1 = df_settle_1.drop('frequent_target')
df_settle1=df_settle_1.toPandas()
#Converting city to tier
tier_info1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/Indian_Cities_by_Tier_I_and_II.csv')
tier_info=tier_info1.toPandas()
tier_info.dropna(subset=['Tier'], inplace=True)
tier_mapper = tier_info[['City', 'Tier']].set_index('City').to_dict()['Tier']    
def tier_mapping(city):
    try: return tier_mapper[city.capitalize()]
    except: return 'Tier-III'
df_settle1['Tier'] = df_settle1['City'].map(tier_mapping)
df_settle1["gender"].replace({"f": "F"}, inplace=True)
df_settle1.drop(['sauda_date','sauda_date1','sauda_date2','party_code','party_code1','party_code2','party_id','cutoff_sauda_date','cumulative_trade_days',
               'target_trade_days','City'], axis=1, inplace=True)

print(df_settle1.shape)
print(df_settle1['settle_target'].value_counts())  
display(df_settle1)



# COMMAND ----------

# DBTITLE 1,Distribution of features
for col in df_settle1.columns:
    print(df_settle1[col].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC #Model building for predicting settle_target

# COMMAND ----------

# DBTITLE 1,train/test split
X_train = df_settle.drop(['settle_target','End_of_Session', 'Cover_order', 'Good_till_cancel', 'Good_till_days', 'Arbitrage_order'], axis=1)
y_train = df_settle['settle_target']

X_test = df_settle1.drop(['settle_target','End_of_Session', 'Cover_order', 'Good_till_cancel', 'Good_till_days', 'Arbitrage_order'], axis=1)
y_test = df_settle1["settle_target"]

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,autoML for settle_target
summary_settle = automl.classify(df_settle, target_col="settle_target", timeout_minutes=30)
model_uri_settle = summary_settle.best_trial.model_path
#help(summary1)
 
# Run inference using the best model
model_set_auto = mlflow.sklearn.load_model(model_uri_settle)
predictions = model_set_auto.predict(X_test)
df_settle1["settle_target_predicted"] = predictions
# Log metrics for the test set
automl_set_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_set_auto, X_test, y_test,prefix="test_")
display(pd.DataFrame([automl_set_test_metrics]))

#Display confusion matrix
sklearn.metrics.plot_confusion_matrix(model_set_auto, X_test, y_test)


# COMMAND ----------

# DBTITLE 1,Performing feature engineering to run models manually 
supported_cols = ["Delivery_cash", "Day_validity", "has_curr_trade", "Brokerage", "Normal_fno", "Bracket_order", "has_mis_trade", "Market_clearing", "T_O", "has_cnc_trade", "riskcategory", "Limit_clearing", "Margin_intraday", "has_fut_idx_trade", "Margin_delivery", "No_group", "Immediate_or_Cancel", "has_opt_idx_trade", "Sell", "incomedetails", "has_comm_trade", "Special_eq", "has_fut_stk_trade", "gender", "occupation", "Buy", "Stop_loss_clearing", "has_opt_stk_trade", "Tier"]

# Define categorical columns
categorical = list(X_train.select_dtypes('object').columns)
print(f"Categorical columns are: {categorical}")

# Define numerical columns
numerical = list(X_train.select_dtypes('float64').columns)
print(f"Numerical columns are: {numerical}")

# Define boolean columns
bool = list(X_train.select_dtypes('bool').columns)
print(f"Boolean columns are: {bool}")

#Boolean columns: For each column, impute missing values and then convert into ones and zeros.
bool_pipeline = Pipeline(steps=[
    ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
    ("imputer1", SimpleImputer(missing_values=None, strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

#Numerical columns: Missing values for numerical columns are imputed with mean for consistency
numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputer2", SimpleImputer(strategy="mean"))
])


#Categorical columns
cat_pipeline = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer([('cat_cols', cat_pipeline, categorical),
                                ('num_cols', numerical_pipeline, numerical),
                                ('bool_cols', bool_pipeline, bool)],remainder="passthrough", sparse_threshold=0)


standardizer = StandardScaler()


preprocessor.fit(X_train)

# Prepare column names
cat_columns = preprocessor.named_transformers_['cat_cols']['encoder'].get_feature_names(categorical)
columns = np.append(cat_columns, numerical)
bool_columns = preprocessor.named_transformers_['bool_cols']['onehot'].get_feature_names(bool)
columns = np.append(columns, bool_columns)
print(columns)


# Inspect training data before and after
print("******************** Training data ********************")
display(X_train)
display(pd.DataFrame(preprocessor.transform(X_train), columns=columns))

# Inspect test data before and after
print("******************** Test data ********************")
display(X_test)
display(pd.DataFrame(preprocessor.transform(X_test), columns=columns))



# COMMAND ----------

# DBTITLE 1, Running LightGBM manually
lgbmc_classifier = LGBMClassifier(
  colsample_bytree=0.5147819785390771,
  lambda_l1=0.11245591123401574,
  lambda_l2=37.22802054235251,
  learning_rate=4.973647345682165,
  max_bin=414,
  max_depth=8,
  min_child_samples=20,
  n_estimators=1199,
  num_leaves=112,
  path_smooth=10.119277938647777,
  subsample=0.7608431340785347,
  random_state=180656403,
)
model_set_gbm = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", lgbmc_classifier),
])

model_set_gbm.fit(X_train, y_train)
predictions = model_set_gbm.predict(X_test)
df_settle1["settle_target_predicted"] = predictions
df_settle1["probability_class_0"] = model_set_gbm.predict_proba(X_test)[:, 0]
df_settle1["probability_class_1"] = model_set_gbm.predict_proba(X_test)[:, 1]
df_settle1["probability_class_2"] = model_set_gbm.predict_proba(X_test)[:, 2]
display(df_settle1)
# Log metrics for the test set
lgbmc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_set_gbm, X_test, y_test, prefix="test_")
# Display the logged metrics
display(pd.DataFrame([lgbmc_test_metrics]))


# COMMAND ----------

# DBTITLE 1,LightGBM classwise evaluation
target_names = ['0', '1', '2']
print(classification_report(y_test, predictions, target_names=target_names, digits=4))
#classwise accuracy
matrix = confusion_matrix(y_test, predictions)
matrix.diagonal()/matrix.sum(axis=1)

# COMMAND ----------

# DBTITLE 1,Light GBM feature importance
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).

    mode = X_train.mode().iloc[0]
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(10, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_test.sample(n=100).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from training set.
    predict = lambda x: model_set_gbm.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model_set_gbm.classes_)

# COMMAND ----------

# DBTITLE 1, Running XGBoost manually
xgbc_classifier = XGBClassifier(
  colsample_bytree=0.5388618260923027,
  learning_rate=1.3266911816246303,
  max_depth=5,
  min_child_weight=7,
  n_estimators=845,
  n_jobs=100,
  subsample=0.32424301245515147,
  verbosity=0,
  random_state=180656403,
)

model_set_xgb = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", xgbc_classifier),
])

model_set_xgb.fit(X_train, y_train)
predictions = model_set_xgb.predict(X_test)
df_settle1["settle_target_predicted"] = predictions
df_settle1["probability_class_0"] = model_set_xgb.predict_proba(X_test)[:, 0]
df_settle1["probability_class_1"] = model_set_xgb.predict_proba(X_test)[:, 1]
df_settle1["probability_class_2"] = model_set_xgb.predict_proba(X_test)[:, 2]
display(df_settle1)
# Log metrics for the test set
xgb_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_set_xgb, X_test, y_test, prefix="test_")
# Display the logged metrics
display(pd.DataFrame([xgb_test_metrics]))


# COMMAND ----------

# DBTITLE 1,XGBoost classwise evaluation
print(classification_report(y_test, predictions, target_names=target_names, digits=4))
#classwise accuracy
matrix = confusion_matrix(y_test, predictions)
matrix.diagonal()/matrix.sum(axis=1)

# COMMAND ----------

# DBTITLE 1,XGBoost feature importance
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).

    mode = X_train.mode().iloc[0]
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(10, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_test.sample(n=100).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from training set.
    predict = lambda x: model_set_xgb.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model_set_xgb.classes_)

# COMMAND ----------

# DBTITLE 1, Running RandomForestClassifier manually
skrf_classifier = RandomForestClassifier(
  bootstrap=False,
  criterion="gini",
  max_depth=12,
  max_features=0.6596812559597282,
  min_samples_leaf=0.012500301023895183,
  min_samples_split=0.04160159682728773,
  n_estimators=49,
  random_state=180656403,
)

model_set_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", skrf_classifier),
])

model_set_rf.fit(X_train, y_train)
predictions = model_set_rf.predict(X_test)
df_settle1["settle_target_predicted"] = predictions
df_settle1["probability_class_0"] = model_set_rf.predict_proba(X_test)[:, 0]
df_settle1["probability_class_1"] = model_set_rf.predict_proba(X_test)[:, 1]
df_settle1["probability_class_2"] = model_set_rf.predict_proba(X_test)[:, 2]
display(df_settle1)
skrf_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_set_rf, X_test, y_test, prefix="test_")
display(pd.DataFrame([skrf_test_metrics]))


# COMMAND ----------

# DBTITLE 1,Randomforest classwise evaluation
print(classification_report(y_test, predictions, target_names=target_names, digits=4))
#classwise accuracy
matrix = confusion_matrix(y_test, predictions)
matrix.diagonal()/matrix.sum(axis=1)

# COMMAND ----------

# DBTITLE 1,Reason for class1 == 0
set(y_test) - set(predictions)

# COMMAND ----------

# DBTITLE 1,Randomforest Feature Importance
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).

    mode = X_train.mode().iloc[0]
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(10, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_test.sample(n=100).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from training set.
    predict = lambda x: model_set_rf.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model_set_rf.classes_)

# COMMAND ----------

# DBTITLE 1, Running DecisionTree manually
skdtc_classifier = DecisionTreeClassifier(
  criterion="entropy",
  max_depth=10,
  max_features=0.45755990848288586,
  min_samples_leaf=0.0828452662159135,
  min_samples_split=0.14297165512232066,
  random_state=180656403,
)

model_set_dt = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", skdtc_classifier),
])

model_set_dt.fit(X_train, y_train)
predictions = model_set_dt.predict(X_test)
df_settle1["settle_target_predicted"] = predictions
df_settle1["probability_class_0"] = model_set_dt.predict_proba(X_test)[:, 0]
df_settle1["probability_class_1"] = model_set_dt.predict_proba(X_test)[:, 1]
df_settle1["probability_class_2"] = model_set_dt.predict_proba(X_test)[:, 2]
display(df_settle1)

skdt_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_set_dt, X_test, y_test, prefix="test_")
display(pd.DataFrame([skdt_test_metrics]))

fig = plt.figure(figsize=(25,20))
#plot_tree(model_set_dt['classifier'],feature_names=list(X_train.columns),class_names=['0','1','2'])
#print(model_set_dt.classes_)
plot_tree(model_set_dt['classifier'],feature_names=columns,class_names=['0','1','2'])


# COMMAND ----------

# DBTITLE 1,Decision Tree classwise evaluation
print(classification_report(y_test, predictions, target_names=target_names, digits=4))
#classwise accuracy
matrix = confusion_matrix(y_test, predictions)
matrix.diagonal()/matrix.sum(axis=1)

# COMMAND ----------

# DBTITLE 1,Reason for class1 == 0
set(y_test) - set(predictions)

# COMMAND ----------

# DBTITLE 1,Decision Tree Feature Importance
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).

    mode = X_train.mode().iloc[0]
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(10, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_test.sample(n=100).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from training set.
    predict = lambda x: model_set_dt.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model_set_dt.classes_)

# COMMAND ----------

# DBTITLE 1, Running Logistic Regression manually
sklr_classifier = LogisticRegression(
  C=22.350922404139133,
  penalty="l2",
  random_state=180656403,
  multi_class="multinomial")

model_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", sklr_classifier),
])

model_lr.fit(X_train, y_train)
predictions = model_lr.predict(X_test)
df_settle1["settle_target_predicted"] = predictions
df_settle1["probability_class_0"] = model_lr.predict_proba(X_test)[:, 0]
df_settle1["probability_class_1"] = model_lr.predict_proba(X_test)[:, 1]
df_settle1["probability_class_2"] = model_lr.predict_proba(X_test)[:, 2]
display(df_settle1)

sklr_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_lr, X_test, y_test, prefix="test_")
display(pd.DataFrame([sklr_test_metrics]))

# COMMAND ----------

# DBTITLE 1,Logistic Regression classwise report
print(classification_report(y_test, predictions, target_names=target_names, digits=4))
#classwise accuracy
matrix = confusion_matrix(y_test, predictions)
matrix.diagonal()/matrix.sum(axis=1)

# COMMAND ----------

# DBTITLE 1,Logistic Regression Feature Importance
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).

    mode = X_train.mode().iloc[0]
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(10, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_test.sample(n=100).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from training set.
    predict = lambda x: model_lr.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model_lr.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Model2 : Using settling customers data to predict if customer can move to the frequent bucket

# COMMAND ----------

# DBTITLE 1,Feature engineering for frequent target train data
#creation of df is common for both settle target and frequent target prediction
df_frequent1 = df.filter( (df['cumulative_trade_days'] == 2) | (df['cumulative_trade_days'] == 3) )
df_frequent1 = df_frequent1.filter((df_frequent1['sauda_date'] <= '2022-02-03'))

df_frequent1 = df_frequent1.drop('settle_target')
df_frequent=df_frequent1.toPandas()

cols = ["has_mis_trade","has_cnc_trade","has_fut_idx_trade","has_fut_stk_trade","has_opt_idx_trade","has_opt_stk_trade","has_curr_trade","has_comm_trade"]
df_frequent[cols] = df_frequent[cols].replace({True: 1, False: 0})    #Changing to integer type so that we can add them
df_frequent_group = df_frequent.groupby('party_code')['has_mis_trade','has_cnc_trade','has_fut_idx_trade','has_fut_stk_trade','has_opt_idx_trade','has_opt_stk_trade','has_curr_trade','has_comm_trade',
    'Brokerage', 'T_O','Day_validity','Immediate_or_Cancel','Good_till_days','Good_till_cancel','End_of_Session','Delivery_cash','Margin_intraday','Margin_delivery','Normal_fno',      'Bracket_order','Arbitrage_order','Cover_order','Market_clearing','Limit_clearing','Stop_loss_clearing','Buy','Sell','No_group','Special_eq'].sum().reset_index()

df_frequent_group[cols] = df_frequent_group[cols].replace({2:1})    
def bool_val(row):
    if row ==1:
        val = True
    else:
        val = False
    return val
for col in cols: 
    df_frequent_group[col] = df_frequent_group[col].apply(bool_val)
    
df_frequent_rem = df_frequent[['party_code','frequent_target','gender','riskcategory','incomedetails','occupation','City']]
df_frequent = pd.merge(df_frequent_group, df_frequent_rem, on='party_code', how='left')
df_frequent = df_frequent.drop_duplicates()
df_frequent[cols] = df_frequent[cols].replace({1: True, 0: False})   #Moving back to boolean type

#Converting city to tier
tier_info1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/Indian_Cities_by_Tier_I_and_II.csv')
tier_info=tier_info1.toPandas()
tier_info.dropna(subset=['Tier'], inplace=True)
tier_mapper = tier_info[['City', 'Tier']].set_index('City').to_dict()['Tier']    
def tier_mapping(city):
    try: return tier_mapper[city.capitalize()]
    except: return 'Tier-III'
df_frequent['Tier'] = df_frequent['City'].map(tier_mapping)
df_frequent.drop(['City','party_code'], axis = 1,inplace=True)
df_frequent["gender"].replace({"f": "F"}, inplace=True)

print(df_frequent.shape)
print(df_frequent['frequent_target'].value_counts())  
display(df_frequent)


# COMMAND ----------

# DBTITLE 1,Distribution of features
for col in df_frequent.columns:
    print(df_frequent[col].value_counts())


# COMMAND ----------

# DBTITLE 1,Feature engineering for frequent target test data
df1_frequent1 = df1.filter( (df1['cumulative_trade_days'] == 2) | (df1['cumulative_trade_days'] == 3) )
df1_frequent1 = df1_frequent1.filter((df1_frequent1['sauda_date'] <= '2022-04-15'))
df1_frequent1 = df1_frequent1.drop('settle_target')
df1_frequent=df1_frequent1.toPandas()

cols = ["has_mis_trade","has_cnc_trade","has_fut_idx_trade","has_fut_stk_trade","has_opt_idx_trade","has_opt_stk_trade","has_curr_trade","has_comm_trade"]
df1_frequent[cols] = df1_frequent[cols].replace({True: 1, False: 0})
df1_frequent_group = df1_frequent.groupby('party_code')['has_mis_trade','has_cnc_trade','has_fut_idx_trade','has_fut_stk_trade','has_opt_idx_trade','has_opt_stk_trade','has_curr_trade','has_comm_trade',
    'Brokerage', 'T_O','Day_validity','Immediate_or_Cancel','Good_till_days','Good_till_cancel','End_of_Session','Delivery_cash','Margin_intraday','Margin_delivery','Normal_fno',      'Bracket_order','Arbitrage_order','Cover_order','Market_clearing','Limit_clearing','Stop_loss_clearing','Buy','Sell','No_group','Special_eq'].sum().reset_index()

df1_frequent_group[cols] = df1_frequent_group[cols].replace({2:1})    

def bool_val(row):
    if row ==1:
        val = True
    else:
        val = False
    return val

for col in cols: 
    df1_frequent_group[col] = df1_frequent_group[col].apply(bool_val)
    
df1_frequent_rem = df1_frequent[['party_code','frequent_target','gender','riskcategory','incomedetails','occupation','City']]
df1_frequent = pd.merge(df1_frequent_group, df1_frequent_rem, on='party_code', how='left')
df1_frequent = df1_frequent.drop_duplicates()

df1_frequent[cols] = df1_frequent[cols].replace({1: True, 0: False})
#Converting city to tier
tier_info1 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/Indian_Cities_by_Tier_I_and_II.csv')
tier_info=tier_info1.toPandas()
tier_info.dropna(subset=['Tier'], inplace=True)
tier_mapper = tier_info[['City', 'Tier']].set_index('City').to_dict()['Tier']    
def tier_mapping(city):
    try: return tier_mapper[city.capitalize()]
    except: return 'Tier-III'
df1_frequent['Tier'] = df1_frequent['City'].map(tier_mapping)
df1_frequent.drop(['City','party_code'], axis = 1,inplace=True)
df1_frequent["gender"].replace({"f": "F"}, inplace=True)


print(df1_frequent.shape)
print(df1_frequent['frequent_target'].value_counts())  
display(df1_frequent)


# COMMAND ----------

# DBTITLE 1,Distribution of features
for col in df1_frequent.columns:
    print(df1_frequent[col].value_counts())

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC #Model building for predicting frequent_target

# COMMAND ----------

# DBTITLE 1,train/test split
X_train = df_frequent.drop(['frequent_target','End_of_Session', 'Cover_order', 'Good_till_cancel', 'Good_till_days', 'Arbitrage_order'], axis=1)
y_train = df_frequent['frequent_target']

X_test = df1_frequent.drop(['frequent_target','End_of_Session', 'Cover_order', 'Good_till_cancel', 'Good_till_days', 'Arbitrage_order'], axis=1)
y_test = df1_frequent["frequent_target"]



# COMMAND ----------

# DBTITLE 1,autoML for frequent_target
summary1 = automl.classify(df_frequent, target_col="frequent_target", timeout_minutes=30)
model_uri1 = summary1.best_trial.model_path
#help(summary1)
# Run inference using the best model
model_fre_auto = mlflow.sklearn.load_model(model_uri1)
predictions = model_fre_auto.predict(X_test)
df1_frequent["frequent_target_predicted"] = predictions
df1_frequent["probability_class_0"] = model_fre_auto.predict_proba(X_test)[:, 0]
df1_frequent["probability_class_1"] = model_fre_auto.predict_proba(X_test)[:, 1]
display(df1_frequent)

# Log metrics for the test set
automl_fre_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_fre_auto, X_test, y_test,prefix="test_")
display(pd.DataFrame([automl_fre_test_metrics]))
#Display confusion matrix
sklearn.metrics.plot_confusion_matrix(model_fre_auto, X_test, y_test)


# COMMAND ----------

# DBTITLE 1,Performing feature engineering to run models manually 
supported_cols = ["Delivery_cash", "Day_validity", "has_curr_trade", "Brokerage", "Normal_fno", "Bracket_order", "has_mis_trade", "Market_clearing", "T_O", "has_cnc_trade", "riskcategory", "Limit_clearing", "Margin_intraday", "has_fut_idx_trade", "Margin_delivery", "No_group", "Immediate_or_Cancel", "has_opt_idx_trade", "Sell", "incomedetails", "has_comm_trade", "Special_eq", "has_fut_stk_trade", "gender", "occupation", "Buy", "Stop_loss_clearing", "has_opt_stk_trade", "Tier"]

# Define categorical columns
categorical = list(X_train.select_dtypes('object').columns)
print(f"Categorical columns are: {categorical}")

# Define numerical columns
numerical = list(X_train.select_dtypes('float64').columns)
print(f"Numerical columns are: {numerical}")

# Define boolean columns
bool = list(X_train.select_dtypes('bool').columns)
print(f"Boolean columns are: {bool}")

#Boolean columns: For each column, impute missing values and then convert into ones and zeros.
bool_pipeline = Pipeline(steps=[
    ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
    ("imputer1", SimpleImputer(missing_values=None, strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

#Numerical columns: Missing values for numerical columns are imputed with mean for consistency
numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputer2", SimpleImputer(strategy="mean"))
])


#Categorical columns
cat_pipeline = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer([('cat_cols', cat_pipeline, categorical),
                                ('num_cols', numerical_pipeline, numerical),
                                ('bool_cols', bool_pipeline, bool)],remainder="passthrough", sparse_threshold=0)


standardizer = StandardScaler()


preprocessor.fit(X_train)

# Prepare column names
cat_columns = preprocessor.named_transformers_['cat_cols']['encoder'].get_feature_names(categorical)
columns = np.append(cat_columns, numerical)
bool_columns = preprocessor.named_transformers_['bool_cols']['onehot'].get_feature_names(bool)
columns = np.append(columns, bool_columns)
print(columns)


# Inspect training data before and after
print("******************** Training data ********************")
display(X_train)
display(pd.DataFrame(preprocessor.transform(X_train), columns=columns))

# Inspect test data before and after
print("******************** Test data ********************")
display(X_test)
display(pd.DataFrame(preprocessor.transform(X_test), columns=columns))




# COMMAND ----------

# DBTITLE 1, Running LightGBM manually
lgbmc_classifier = LGBMClassifier(
  colsample_bytree=0.4665051886791723,
  lambda_l1=3.6772653198259615,
  lambda_l2=3.86341994827281,
  learning_rate=0.6787555528928094,
  max_bin=339,
  max_depth=7,
  min_child_samples=97,
  n_estimators=9,
  num_leaves=195,
  path_smooth=79.2942499229715,
  subsample=0.7299193995713901,
  random_state=522987681,
)

model_lgbmc = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", lgbmc_classifier),
])

model_lgbmc.fit(X_train, y_train)


# Log metrics for the test set
lgbmc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_lgbmc, X_test, y_test, prefix="test_")

# Display the logged metrics
display(pd.DataFrame([lgbmc_test_metrics]))


predictions = model_lgbmc.predict(X_test)
df1_frequent["frequent_target_predicted"] = predictions
df1_frequent["probability_class_0"] = model_lgbmc.predict_proba(X_test)[:, 0]
df1_frequent["probability_class_1"] = model_lgbmc.predict_proba(X_test)[:, 1]
display(df1_frequent)




# COMMAND ----------

# DBTITLE 1,Feature importance for LightGBM
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).

    mode = X_train.mode().iloc[0]
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(10, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_test.sample(n=100).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from training set.
    predict = lambda x: model_lgbmc.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model_lgbmc.classes_)

# COMMAND ----------

# DBTITLE 1, Running RandomForestClassifier manually
skrf_classifier = RandomForestClassifier(
  bootstrap=False,
  criterion="gini",
  max_depth=12,
  max_features=0.6764919267775885,
  min_samples_leaf=0.006781566553524826,
  min_samples_split=0.021836908843176295,
  n_estimators=103,
  random_state=522987681,
)

model_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", skrf_classifier),
])

model_rf.fit(X_train, y_train)
# Log metrics for the test set
skrf_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_rf, X_test, y_test, prefix="test_")

# Display the logged metrics
display(pd.DataFrame([skrf_test_metrics]))


predictions = model_rf.predict(X_test)
df1_frequent["frequent_target_predicted"] = predictions
df1_frequent["probability_class_0"] = model_rf.predict_proba(X_test)[:, 0]
df1_frequent["probability_class_1"] = model_rf.predict_proba(X_test)[:, 1]
display(df1_frequent)


# COMMAND ----------

# DBTITLE 1,Feature importance RandomForestClassifier
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).

    mode = X_train.mode().iloc[0]
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(10, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_test.sample(n=100).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from training set.
    predict = lambda x: model_rf.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model_rf.classes_)

# COMMAND ----------

# DBTITLE 1, Running DecisionTree manually
skdtc_classifier = DecisionTreeClassifier(
  criterion="entropy",
  max_depth=4,
  max_features=0.4529938625120228,
  min_samples_leaf=0.0776888725865643,
  min_samples_split=0.19779409765599099,
  random_state=522987681,
)

model_dt = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", skdtc_classifier),
])

model_dt.fit(X_train, y_train)
skdtc_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_dt, X_test, y_test, prefix="test_")
display(pd.DataFrame([skdtc_test_metrics]))



predictions = model_dt.predict(X_test)
df1_frequent["frequent_target_predicted"] = predictions
df1_frequent["probability_class_0"] = model_dt.predict_proba(X_test)[:, 0]
df1_frequent["probability_class_1"] = model_dt.predict_proba(X_test)[:, 1]
display(df1_frequent)

fig = plt.figure(figsize=(25,20))
plot_tree(model_dt['classifier'],feature_names=columns,class_names=['0','1'])

#print(model_dt.classes_)

# COMMAND ----------

# DBTITLE 1,Feature importance decision tree
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).

    mode = X_train.mode().iloc[0]
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(10, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_test.sample(n=100).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from training set.
    predict = lambda x: model_dt.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model_dt.classes_)

# COMMAND ----------

# DBTITLE 1, Running Logistic Regression manually
sklr_classifier = LogisticRegression(
  C=65.56841523483085,
  penalty="l2",
  random_state=522987681,
)

model_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("classifier", sklr_classifier),
])

model_lr.fit(X_train, y_train)
sklr_test_metrics = mlflow.sklearn.eval_and_log_metrics(model_lr, X_test, y_test, prefix="test_")
display(pd.DataFrame([sklr_test_metrics]))

predictions = model_lr.predict(X_test)
df1_frequent["frequent_target_predicted"] = predictions
df1_frequent["probability_class_0"] = model_lr.predict_proba(X_test)[:, 0]
df1_frequent["probability_class_1"] = model_lr.predict_proba(X_test)[:, 1]
display(df1_frequent)


# COMMAND ----------

# DBTITLE 1,Feature importance logistic regression
shap_enabled = True

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).

    mode = X_train.mode().iloc[0]
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(10, len(X_train.index))).fillna(mode)

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_test.sample(n=100).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the example from training set.
    predict = lambda x: model_lr.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example, class_names=model_lr.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC **Reference**
# MAGIC 
# MAGIC https://docs.databricks.com/_static/notebooks/machine-learning/automl-classification-example.html
# MAGIC 
# MAGIC https://www.analyticsvidhya.com/blog/2021/05/understanding-column-transformer-and-machine-learning-pipelines/
# MAGIC 
# MAGIC https://towardsdatascience.com/pipeline-columntransformer-and-featureunion-explained-f5491f815f
