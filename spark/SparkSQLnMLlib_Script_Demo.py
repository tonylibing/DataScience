### -------------------------------------------------- prepare data part --------------------------------------
### ### spark version
### #u'2.0.0.2.5.3.0-37'
###
### train dataset: scp /cygdrive/c/Users/peipe/Desktop/MassData/logdata_10.172.204.245_2017032218 logdata_toutiao.json tel:~/
### test dataset:  scp /cygdrive/c/Users/peipe/Desktop/MassData/logdata_10.26.109.237_2017032218.gz tel:~/
### gunzip logdata_10.26.109.237_2017032218.gz



### ### prepare files on hadoop
### ### copy data from local to hdfs
### hadoop fs -ls /user/hanpeipei
### hadoop fs -copyFromLocal /home/hanpeipei/logdata_toutiao.json   /user/hanpeipei/MassData
##### hadoop fs -copyFromLocal /home/hanpeipei/IP.zip   /user/hanpeipei/MassData
##### hadoop fs -copyFromLocal /home/hanpeipei/unidecode.zip   /user/hanpeipei/MassData
### hadoop fs -ls /user/hanpeipei/MassData



#-------------------------------------------------- data wrangling part--------------------------------------------
from pyspark import SparkConf,SparkContext,SQLContext
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

sqlContext.setConf("spark.network.timeout", "600s")
sqlContext.getConf("spark.network.timeout")

###load data on local 
df = sqlContext.read.json("/home/hanpeipei/MassData/logdata_toutiao.json")       #16096027
df = sqlContext.read.json("/home/hanpeipei/MassData/logdata_toutiao_test.json")  #13631640
###load data on cluster #df = sqlContext.read.json("/user/hanpeipei/MassData/logdata_toutiao.json")#16096027

BidRequestDate = df.select(df.flag,df.bid.alias("BR_bid"),"category","content_categories","excluded_ad_category","excluded_click_through_url","page_session_id","user_agent").filter(df.flag == "bidRequestDate")
BidRequestDate = BidRequestDate.withColumnRenamed("flag","BR_flag")
#3776365

BidUdamMap = df.select(df.flag,df.bid.alias("BU_bid"),df.adCategoryId,df.adId,df.bidPrice,df.commission,df.isOwnUrl,df.materialId,df.mediaUrl,df.originalUrl,df.planId,df.uid,df.ip,df.tid
,df.time.alias("BU_time"),df.timestamp.alias("BU_timestamp")).filter(df.flag == "BidUdamMap")
BidUdamMap = BidUdamMap.withColumnRenamed("flag","BU_flag")
#count 3776365


ClickNotice = df.select(df.flag,df.bid.alias("CN_bid"),df.timestamp.alias("CN_timestamp")).filter(df.flag == "ClickNotice")
ClickNotice = ClickNotice.groupBy(ClickNotice["CN_bid"]).count()
ClickNotice = ClickNotice.dropDuplicates()
ClickNotice =  ClickNotice.withColumn("clickThroughFlag",lit(1))
# count 130

Device = df.select(df.flag,df.bid.alias("Dev_bid"),df.device_pixel_ratio,df.device_size,df.os,df.os_version,df.platform).filter(df.flag == "device")
Device = Device.withColumnRenamed("flag","Dev_flag")
#202189

Adzinfo = df.select(df.flag,df.bid.alias("Adz_bid"),df.allowed_creative_level,df.excluded_filter,df.min_cpm_price,df.pid,df.publisher_filter_id,df.size,df.view_screen,df.view_type).filter(df.flag == "adzinfo")
Adzinfo = Adzinfo.withColumnRenamed("flag","Adz_flag")
#3776365

#Usermonitor = df.select("_corrupt_record","flag").filter("flag is NULL")
#Usermonitor = Usermonitor.select(Usermonitor._corrupt_record.substr(26,32).alias("UM_bid"))
#3574176

#Join tables
df1=BidRequestDate.join(BidUdamMap, BidRequestDate.BR_bid == BidUdamMap.BU_bid, "left_outer")
df1=df1.join(WinNotice, df1.BR_bid == WinNotice.WN_bid, "left_outer")
df1=df1.join(ClickNotice, df1.BR_bid == ClickNotice.CN_bid, "left_outer")
df1=df1.join(Device, df1.BR_bid == Device.Dev_bid, "left_outer")
df1=df1.join(Adzinfo, df1.BR_bid == Adzinfo.Adz_bid, "left_outer")

#Get a copy
df1_cp = df1
#train dataset 3776556
#test dataset  3190061

#GC overhead limit exceeded

### convert ip to location info like country, city, county name
# cp IP.zip to hadoop local
# ip function download address  
# http://www.ipip.net/download.html
# $ scp /cygdrive/c/Users/peipe/Desktop/MassData/ip-search/17monip-master/IP.zip tel:~/
# after cp IP.zip to hadoop use "unzip IP.zip" to uncompress the files


### specify the module ip.py doc directory befor import module ip.py
### specify the module unidecode.zip doc directory befor import module unidecode
sc2 = sc.addPyFile("/home/hanpeipei/IP/ip.py")
sc2 = sc.addPyFile("/home/hanpeipei/IP/17monipdb.dat")
sc2 = sc.addPyFile("/home/hanpeipei/unidecode.zip")

import ip
from unidecode import unidecode 

### example
#	location = ip.find("59.151.24.1")
#	location = unidecode(location)
#	location = location.split("\t")
#	country = location[0]
#	city = location[1]
#	county = location[2]
#	print country,city,county

#User defined function for calculating the location of a clientIP
#"i" used to select which info in location like country, city, county

def location(clientIP,i):
	if clientIP != None:
		if ip.find(clientIP) != None:
			if len(unidecode(ip.find(clientIP)).split("\t")) == 3:
				location = unidecode(ip.find(clientIP)).split("\t")
				city = location[i]
			else:
				city = "unknown"
		else:
			city = "unknown"
	else:
		city = "unknown"
	return city
	
#Define UDF for calculating the length of the diagonal
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, DateType, TimestampType
from pyspark.sql.functions import udf  
locationCvt = udf(location, StringType())	

#Append the length of the diagonal as a new column to the original data frame 
df1 = df1.withColumn("country", locationCvt(df1['clientIP'],lit(0)))
df1 = df1.withColumn("city",    locationCvt(df1['clientIP'],lit(1)))
df1 = df1.withColumn("county",  locationCvt(df1['clientIP'],lit(2)))

#create view df0 from dataframe df1 in order to use the sql command 
df1.createOrReplaceTempView("df0")

#spark 2.0.0 sqlContext.sql
#df2 = sqlContext.sql(""")
#  spark 2.1 spark.sql
df2 = sqlContext.sql("""
select 
	    case when clickThroughFlag == 1 then '1' else '0' end as clickThroughFlag
	   ,case when allowed_creative_level is null then 'unknown' else allowed_creative_level end as allowed_creative_level
	   ,cast(case when bidPrice            is null then 0 else bidPrice end  as integer )as bidPrice
	   
	   ,case when category is null then 'unknown' 
	         when category not in ('41801','40102','40501','40401','41805','41802','41706','42901','40901','40101'
                                  ,'42401','40301','41505','40201','41502','42201','42101','41003') then 'others'
             else category end as category
	 --  ,case when clientIP is null then 'unknown' else clientIP end as clientIP
	   ,country 
	   
	   ,case when city not in ('unknown' ,'Yan Dong' ,'Zhe Jiang' ,'Jiang Su' ,'Shang Hai' ,'Si Chuan' ,'Shan Dong' ,'Shan Xi' ,'Hu Nan'   
								,'Fu Jian' ,'Xin Jiang' ,'Jiang Xi' ,'Hu Bei' ,'Bei Jing' ,'Yun Nan' ,'Gui Zhou' ,'Zhong Qing' ,'An Hui' ,'Yan Xi' ,'Hai Nan' ,'He Nan' ,'Tian Jin' ,'Zhu Xia' ,'Nei Meng Gu' ,'Qing Hai' ,'Liao Zhu' ,'Gan Su' ,'Xi Cang' ,'He Bei' ,'Ji Lin' ,'Hei Long Jiang' ) then 'others'
		     else city end as city
			 
	   ,case when county not in ('unknown','Shang Hai','Yan Zhou' ,'Shen Zhen' ,'Hang Zhou' ,'Dong Wan' ,'Su Zhou' ,'Nan Jing' ,'Fo Shan' ,
								'Zhu Bo' ,'Wen Zhou' ,'Tai Zhou' ,'Zhong Shan' ,'Jin Hua' ,'Wu Xi' ,'Jia Xing' ,'Shao Xing' ,'Hui Zhou' ,'Chang Zhou' ,'Xu Zhou' ,'Jiang Men' ,'Zhu Hai' ,'Hu Zhou' ,'Nan Tong' ,'Yang Zhou' ,'Shan Tou' ,'Lian Yun Gang' ,'Zhen Jiang' ,'Zhan Jiang' ,'Zhao Qing' ,'Yan Cheng' ,'Jie Yang' ) then 'others'
	         else county end as county
			 
	   ,case when commission is null then 0 else cast( commission as float) end  as commission
	   
	   
	   ,case when content_categories[0]["confidence_level"] is null then 'unknown' 
			 when content_categories[0]["confidence_level"] not in ('unknown' ,'950' ,'999' ,'998' ,'997' ,'996' ,'993' ,'995' ,'994' ,'948' ,'976' ,'977' ) then 'others'  
			 else content_categories[0]["confidence_level"] end as content_categories00
			 
	   ,case when content_categories[0]["id"] is null then 'unknown' 
			 when content_categories[0]["id"] not in ('unknown' ,'823' ,'80603' ,'816' ,'81307' ,'81215' ,'82303' ,'82204' ,'806' ,'80608' ,'82208' ,'82504' ,'818' ,'81501' ,'802' ,'826' ,'80601' ,'813' ,'82003' ,'81201' ,'81202' ,'81401' ,'82202' ,'80606' ,'80602' ,'808' ,'81823' ) then 'others'  
			 else content_categories[0]["id"] end as content_categories01
			 
	   ,case when content_categories[1]["confidence_level"] is null then 'unknown' 
	         when content_categories[1]["confidence_level"] not in ('unknown' ,'950' ,'999' ,'997' ,'998' ,'993' ,'996' ,'969' ,'911' ,'928' ,'992' ,'991' ,'968' ,'929' ,'985' ,'970' ,'995' ,'967' ) then 'others'  
	         else content_categories[1]["confidence_level"] end as content_categories10
			 
	   ,case when content_categories[1]["id"] is null then 'unknown' 
			 when content_categories[1]["id"] not in ('unknown' ,'806' ,'813' ,'825' ,'822' ,'815' ) then 'others' 
	         else content_categories[1]["id"] end as content_categories11
	  
	  
	   ,case when device_pixel_ratio  is null then 0 else device_pixel_ratio end  as device_pixel_ratio
	   ,case when excluded_filter[0] is null then 'unknown' else excluded_filter[0] end as excluded_filter
	   ,case when isOwnUrl is null then 'unknown' else isOwnUrl end as isOwnUrl
	   ,cast (case when min_cpm_price       is null then 0 else min_cpm_price end as double) as min_cpm_price
	   ,case when priceReal           is null then 0 else priceReal end  as priceReal
	 --  ,case when mediaUrl is null then 'unknown' else mediaUrl end as mediaUrl
	   ,case when os       is null then 'unknown' else os end as os
	   
	   ,case when os_version is null then 'unknown' 
	         when os_version not in ('unknown' ,'10.2' ,'5.1' ,'6' ,'4.44' ,'6.01' ,'5.11' ,'4.42' ,'7' ,'9.3' ,'5.02') then 'others'
			 else os_version end as os_version
			 
	   ,case when platform is null then 'unknown' 
	         when platform not in ('unknown' ,'android' ,'iphone','ipad','winphone','ipod') then 'others'
			 else platform end as platform
			 
	   ,cast( case when substr(size,1,locate('x',size)-1) is null then 0 else substr(size,1,locate('x',size)-1)  end as integer ) as size_len
	   ,cast( case when substr(size,locate('x',size)+1) is null then 0 else substr(size,locate('x',size)+1) end  as integer) as size_wid
	   ,case when view_type[0] is null then 'unknown' else view_type[0] end as view_type
     --  ,case when cast(BU_timestamp as date) is null then 'unknown' else cast(BU_timestamp as date) end  as date
       ,case when substr(BU_timestamp,12,2)  is null then 'unknown' else substr(BU_timestamp,12,2)  end as hour 
	   ,case when date_format(cast(BU_timestamp as date),'u') is null then 'unknown' else date_format(cast(BU_timestamp as date),'u') end as weekday
	   
	   ,cast( case when substr(device_size,1,locate('x',device_size)-1) is null then 0 else substr(device_size,locate('x',device_size)+1) end as integer) as device_size_len
	   ,cast( case when substr(device_size,locate('x',device_size)+1) is null then 0 else substr(device_size,locate('x',device_size)+1) end as integer) as device_size_wid
	   
	   ,case when excluded_click_through_url[0] is null then 'unknown' 
	         when excluded_click_through_url[0] not in ('unknown' ,'9377a.com' ,'anjuke.com' ,'dell.com.cn' ,'kejet.net' ,'chenghuitong.net' ,'17maib.com' ,'feiniu.cn' ,'bd.kai-ying.com' ,'5399.com' ,'vdax.youzu.com/' ,'airbnb.com' ,'vdax.uuzu.com' ,'renren.com' ,'qq.com' ,'vdax.youzu.com' ,'bl.com' ,'chaoke.com' ,'aliyun.com' ,'t.dahei.com' ,'114.64.245.112' ,'7road.com' ,'vdax.uuzu.com/' ,'bb.ztgame.com/static/live/' ) then 'others'
	         else excluded_click_through_url[0] end as excluded_click_through_url0
			 
	   ,case when excluded_click_through_url[1] is null then 'unknown' 
			 when excluded_click_through_url[1] not in ('unknown' ,'czpanshi.com' ,'cheshi.com' ,'intel.cn' ,'wisemedia.cn' ,'coopertire.com.cn' ,'feiniu.com' ,'bdtg.4366.com' ,'45993.com' ,'7road.com' ,'37wan.com' ,'dahei.com' ,'www.xyxy01.com' ,'weibo.com' ,'sina.com' ,'blemall.com' ,'lillycare.com.cn' ,'www.aliyun.com' ,'114.64.245.112/t2' ,'95k.com' ) then 'others' 
			 else excluded_click_through_url[1] end as excluded_click_through_url1
			 
	   ,case when excluded_click_through_url[2] is null then 'unknown' 
			 when excluded_click_through_url[2] not in ('unknown' ,'shop.letv.com' ,'hanergy.com' ,'nuomi.com' ,'metao.com' ,'jd.com' ,'bdtg.9377a.com' ,'emoney.cn' ,'9377z.com' ,'51.com' ,'moodoo.com.cn' ,'sohu.com' ,'feiniu.cn' ,'ssofair.com' ) then 'others' 
			 else excluded_click_through_url[2] end as excluded_click_through_url2
			 
	   ,case when excluded_click_through_url[3] is null then 'unknown' 
			 when excluded_click_through_url[3] not in ('unknown' ,'hanergyshop.com' ,'odg.dell-brand.com' ,'suning.com' ,'bdtg.9377s.com' ,'ladysite.net' ,'aoshitang.com' ,'bid.syhccs.com' ,'game2.cn' ,'yiguo.com' ,'feiniu.com' ,'tg.51.com' ,'vda.9787.com' ,'www.benlai.com' ) then 'others' 
			 else excluded_click_through_url[3] end as excluded_click_through_url3
			 
	   ,case when excluded_click_through_url[4] is null then 'unknown' 
			 when excluded_click_through_url[4] not in ('unknown' ,'kaola.com' ,'vancl.com' ,'vip.com' ,'dev.tg.youxi.com' ,'ppdai.com' ,'bid.syhccs.com' ,'game2.cn' ,'gome.com.cn' ) then 'others' 
			 else excluded_click_through_url[4] end as excluded_click_through_url4
	   
	   ,case when excluded_ad_category[0] is null then 'unknown' 
			 when excluded_ad_category[0] not in ('unknown' ,'60404' ,'72701' ,'60206' ,'62601' ,'60205' ,'60505' ,'61101' ,'60301' ,'62701' ,'62402' ,'61103' ,'72117' ,'60201' 	) then 'others' 
			 else excluded_ad_category[0] end as excluded_ad_category0 
			 
	   ,case when excluded_ad_category[1] is null then 'unknown' 
			 when excluded_ad_category[1] not in ('unknown' ,'60407' ,'72702' ,'62602' ,'60505' ,'60206' ,'60301' ,'60404' ,'60515' ,'61102' ,'60302' ,'62702' ,'61106' ,'71514' ) then 'others' 
			 else excluded_ad_category[1] end as excluded_ad_category1
			 
	   ,case when excluded_ad_category[2] is null then 'unknown' 
			 when excluded_ad_category[2] not in ('unknown' ,'60505' ,'72703' ,'62603' ,'61103' ,'60511' ,'60301' ,'60302' ,'60407' ,'60417' ,'60303' ,'62703' ,'72701' ,'72702' ,'62116' ,'60203' ,'62402' ) then 'others' 
			 else excluded_ad_category[2] end as excluded_ad_category2
			 
	   ,case when excluded_ad_category[3] is null then 'unknown' 
		     when excluded_ad_category[3] not in ('unknown' ,'60508' ,'60515' ,'62604' ,'60302' ,'60303' ,'60505' ,'61104' ,'61106' ,'60611' ,'60304' ,'62704' ,'72702' ,'62402' ,'72703' ,'60204' ,'71514' ,'62605' ) then 'others' 
			 else excluded_ad_category[3] end as excluded_ad_category3 
			 
	   ,case when excluded_ad_category[4] is null then 'unknown' 
	         when excluded_ad_category[4] not in ('unknown' ,'60511' ,'62605' ,'61104' ,'60303' ,'60304' ,'61105' ,'62116' ,'61101' ,'61107' ,'60305' ,'62705' ,'62601' ,'72703' ,'60205' ,'71515' ) then 'others' 
			 else excluded_ad_category[4] end as excluded_ad_category4
	   
	   from df0 """ )

#df2 train dataset count 3776556	   
#df2 test dataset count  3190061

df2.write.json("/home/hanpeipei/df2.json")
df3 = sqlContext.read.json("/home/hanpeipei/df2.json")

df2.write.json("/home/hanpeipei/df2.json")
df3 = sqlContext.read.json("/home/hanpeipei/df2_test.json")   #3190061

### ------------------------------------------------- Machine Learning Part-----------------------------------------------------
# Feature Extraction 
# Extract features tools in with pyspark.ml.feature						

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

numerical_cols = [
 'bidPrice'
,'commission'
,'device_pixel_ratio'
,'priceReal'
,'size_len'
,'size_wid'
,'device_size_len'
,'device_size_wid'
,'min_cpm_price'
]

categorical_cols = [
'allowed_creative_level'
,'category' 
,'country'
,'city'
,'county'
,'content_categories00'  
,'content_categories01'
,'content_categories10'
,'content_categories11'
,'excluded_filter'
,'isOwnUrl'
,'os'
,'os_version'
,'platform'
,'view_type'
,'hour'
,'weekday'
,'excluded_click_through_url0'
,'excluded_click_through_url1'
,'excluded_click_through_url2'
,'excluded_click_through_url3'
,'excluded_click_through_url4'
,'excluded_ad_category0'
,'excluded_ad_category1'
,'excluded_ad_category2'
,'excluded_ad_category3'
,'excluded_ad_category4'
]

label_indexer = StringIndexer(inputCol = 'clickThroughFlag', outputCol = 'label',handleInvalid='skip')

### http://spark.apache.org/docs/latest/api/python/pyspark.ml.html?highlight=handleinvalid#pyspark.ml.feature.StringIndexer.handleInvalid
### https://stackoverflow.com/questions/34681534/spark-ml-stringindexer-handling-unseen-labels

# Turn category fields into indexes
cat_indexer = []
categorical_indexed_cols = []
for column in categorical_cols:
	cat_indexer.append(StringIndexer(inputCol=column,outputCol=column + "_index",handleInvalid='skip'))
	categorical_indexed_cols.append(column + "_index")

    
assembler = VectorAssembler(inputCols = categorical_indexed_cols + numerical_cols, outputCol = 'features')

#----------------------------------------------------# Model Trainning---------------------------------------------

#---------------------------------------------------# DecisionTreeClassifier --------------------------------------
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

classifier = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'features', maxBins=50)

pipeline = Pipeline(stages=[
cat_indexer[0],cat_indexer[1]
,cat_indexer[2],cat_indexer[3]
,cat_indexer[4],cat_indexer[5]
,cat_indexer[6],cat_indexer[7]
,cat_indexer[8],cat_indexer[9]
,cat_indexer[10],cat_indexer[11]
,cat_indexer[12],cat_indexer[13]
,cat_indexer[14],cat_indexer[15]
,cat_indexer[16],cat_indexer[17]
,cat_indexer[18],cat_indexer[19]
,cat_indexer[20],cat_indexer[21]
,cat_indexer[22],cat_indexer[23]
,cat_indexer[24],cat_indexer[25]
,cat_indexer[26]
,label_indexer, assembler, classifier])

### cross validation 
### same error as before
# Failed to execute user defined #
# function anonfun$4: (string) => double)
#grid = ParamGridBuilder().build()
#crossval = CrossValidator(estimator=pipeline,
#                          estimatorParamMaps=grid,
#                          evaluator=BinaryClassificationEvaluator(),
#                          numFolds=5)  # use 5/10 folds in practice
# Run cross-validation, and choose the best set of parameters.
#model = crossval.fit(df3)
#predictions = model.transform(df3)

#(train, test) = df3.randomSplit([0.8, 0.2])
#model = pipeline.fit(train)
#predictions = model.transform(test)

df3_train = sqlContext.read.json("/home/hanpeipei/df2.json")
df3_test = sqlContext.read.json("/home/hanpeipei/df2_test.json")

train, test = df3_train,df3_test
model = pipeline.fit(train)
predictions = model.transform(test)


###### debug referrence url  https://github.com/adornes/spark_python_ml_examples
#If you fit a StringIndexer over the training dataset and afterwards, when the pipeline is used to predict an outcome over another #dataset (validation, test, etc.), it faces some unseen category, then it will fail and raise the error: org.apache.spark.SparkException:
#Failed to execute user defined function($anonfun$4: (string) => double) ... Caused by: org.apache.spark.SparkException: Unseen label: #XYZ ... at org.apache.spark.ml.feature.StringIndexerModel. This is the reason why the scripts' code fits the StringIndexer #transformations over a union of original data from train.csv and test.csv, bypassing the sampling and split parts.


#DecisionTree Model Evaluation
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s." % (auroc, aupr)	
### sample 3873 result
#'The AUROC is 0.629427792916 and the AUPR is 0.144590564558.'
### entire data train  
#'The AUROC is 0.324789221125 and the AUPR is 0.0714374659345.'
### data test result
"The AUROC is %s and the AUPR is %s." % (auroc, aupr)
'The AUROC is 0.57472764062 and the AUPR is 0.106091167835.'



