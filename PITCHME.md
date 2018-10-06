# Spark ML

## M.1 학습내용

### M.1.1 목표

* 사례별로 데이터를 분석하여 분류, 군집, 추천, 예측
---
### M.1.2 목차

* M.1 IPython Notebook에서 SparkSession 생성하기
* M.2 사례 
* M.2.1 구조적 데이터
* M.2.2 텍스트 데이터
* M.2.3 LDA
* M.2.4 영화추천
* M.2.5 분석 절차
---

### M.1.3 문제 

* 문제 M-1: Titanic case
* 문제 M-2: Kaggle Twitter US Airline Sentiment
* 문제 M-3: Spark MLib Decision Tree for kddcup99
* 문제 M-4: LDA 
* 문제 M-5: Spark MLib movie recommendation 사례
* 문제 M-6: Ethereum
* 문제 M-7: Spark Streaming
* 문제 M-8: GraphX
* spark-submit (self-contained app in quick-start 참조) 
---

## M.1 IPython Notebook에서 SparkSession 생성하기

Jupyter Notebook에서 Spark를 사용하려면 몇 가지 설정이 필요하다.
Spark 실행에 필요한 라이브러리를 경로에서 찾을 수 있게 설정해야 한다.
아래와 같이 ```sys.path.insert()``` 함수를 사용하면 라이브러리를 Python 경로에 추가할 수 있다.


```python
import os
import sys 
os.environ["SPARK_HOME"]=os.path.join(os.environ['HOME'],'Downloads','spark-2.0.0-bin-hadoop2.7')
os.environ["PYLIB"]=os.path.join(os.environ["SPARK_HOME"],'python','lib')
sys.path.insert(0,os.path.join(os.environ["PYLIB"],'py4j-0.10.1-src.zip'))
sys.path.insert(0,os.path.join(os.environ["PYLIB"],'pyspark.zip'))
```
---
SparkSession에 필요한 설정을 넣어서 만들어 준다.
SparkSession은 getOrCreate()로 이미 만들어져 있으면 현재 것을, 없으면 생성하는 방식을 취하고 있다.


```python
import pyspark
myConf=pyspark.SparkConf()
spark = pyspark.sql.SparkSession.builder\
    .master("local")\
    .appName("myApp")\
    .config(conf=myConf)\
    .getOrCreate()
```
---
Mongo DB를 사용하기 위해서는 실행시켜 준비해 놓는다.  Mongo daemon은 다음과
같이 실행한다.
```
mongod --dbpath ./data/
```
---
## M.2 사례

다음 사례를 기계학습으로 풀어보기로 한다.

### M.2.1 구조적 데이터

Titanic case, kddcup99

### M.2.2 텍스트 데이터

Kaggle Twitter US Airline Sentiment

### M.2.3 LDA

같은 주제로 단어들을 묶어 토픽모델을 만든다.

### M.2.4 영화추천

자신의 선호에 따른 영화를 추천하는 기계학습이다.
---
### M.2.5 분석 절차

* 1 단계: 데이터 수집
* 2 단계: 데이터 변환 - 탐색 및 ETL
* 3 단계: 모델링
* 4 단계: 예측
* 5 단계: 평가 - 평가 및 모델의 개선

---
## 문제 M-1: Titanic case

* 1912년 4월 15일 Titanic 유람선 사고, 2224명의 승객 및 선원 가운데 1502명이 사망
* 사망 여부의 2진 분류
* Kaggle에 공개된 데이터
---
### M-1.1 데이터 수집

파일 | 설명
-----|-----
train.csv | 훈련 파일, 'Survived' 열의 값을 label로 사용한다.
test.csv | 테스트 파일, 'Survived' 열의 값을 예측해야 한다.
gender_submission.csv | 예측 결과 제출 파일 예제


```text
!ls data/kaggle/titanic/
```

    gender_submission.csv  test.csv  train.csv

---
* 'train.csv'와 'test.csv'를 합쳐서 하나의 파일로 만든다.


```python
_trainDf = spark.read.format('com.databricks.spark.csv')\
    .options(header='true', inferschema='true')\
    .load(os.path.join("data","kaggle","titanic","train.csv"))
_trainDf.take(1)


    [Row(PassengerId=1, Survived=0, Pclass=3, Name=u'Braund, Mr. Owen Harris', Sex=u'male', Age=22.0, SibSp=1, Parch=0, Ticket=u'A/5 21171', Fare=7.25, Cabin=u'', Embarked=u'S')]
```

---


```python
_testDf = spark.read.format('com.databricks.spark.csv')\
    .options(header='true', inferschema='true')\
    .load(os.path.join("data","kaggle","titanic","test.csv"))
_testDf.take(1)




    [Row(PassengerId=892, Pclass=3, Name=u'Kelly, Mr. James', Sex=u'male', Age=34.5, SibSp=0, Parch=0, Ticket=u'330911', Fare=7.8292, Cabin=u'', Embarked=u'Q')]

```

---
* 'Survived'는 'train.csv'에는 있으나, 'test.csv'에는 없다. 따라서 임의의 수 99를 넣는다.
    * pyspark.sql.functions.lit(col) 컬럼에 값을 넣어 열을 생성하는 기능



```python
from pyspark.sql.functions import lit, col
_trainDf = _trainDf.withColumn('testOrtrain',lit('train'))
_testDf = _testDf.withColumn('testOrtrain',lit('test'))
_testDf = _testDf.withColumn('Survived',lit(99))
```
---
* union
    * DataFrame을 합치는 기능
    * 두 DataFrame의 컬럼 수와 데이터타잎이 일치해야 한다. 순서가 다르더라도 그냥 합치는 것에 주의한다.

* Sql의 union은 컬럼명을 고려하지 않고 컬럼수만 동일하면 합쳐준다.
    * 별도 추가된 'Survived'열이 맨 뒤에 위치하게 되고, 다른 열과 합쳐지게 된다.
    * 컬럼명을 모두 적어주어 해결한다.
---

```python
_trainDf.printSchema()

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
     |-- testOrtrain: string (nullable = false)
    
```
---

```python
_testDf.printSchema()

    root
     |-- PassengerId: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
     |-- testOrtrain: string (nullable = false)
     |-- Survived: integer (nullable = false)
    
```
---


```python
for c in _trainDf.columns:
    print c,
```
---

```python
df=_trainDf.select('PassengerId','Survived','Pclass','Name','Sex','Age',\
                   'SibSp','Parch','Ticket','Fare','Cabin','Embarked','testOrtrain')\
            .union(_testDf.select('PassengerId','Survived','Pclass','Name','Sex','Age',\
                   'SibSp','Parch','Ticket','Fare','Cabin','Embarked','testOrtrain'))
```
---
* test 또는 train선택 filter(condition)
* where() is an alias for filter().

```python
df.select('testOrtrain','Survived','Name')\
    .filter(df['testOrtrain']=='test').show(10)

    +-----------+--------+--------------------+
    |testOrtrain|Survived|                Name|
    +-----------+--------+--------------------+
    |       test|      99|    Kelly, Mr. James|
    |       test|      99|Wilkes, Mrs. Jame...|
    |       test|      99|Myles, Mr. Thomas...|
    |       test|      99|    Wirz, Mr. Albert|
    |       test|      99|Hirvonen, Mrs. Al...|
    |       test|      99|Svensson, Mr. Joh...|
    |       test|      99|Connolly, Miss. Kate|
    |       test|      99|Caldwell, Mr. Alb...|
    |       test|      99|Abrahim, Mrs. Jos...|
    |       test|      99|Davies, Mr. John ...|
    +-----------+--------+--------------------+
    only showing top 10 rows
```
    
---


```python
df.groupBy(df.testOrtrain).count().show()

    +-----------+-----+
    |testOrtrain|count|
    +-----------+-----+
    |      train|  891|
    |       test|  418|
    +-----------+-----+
```
    
---

### M-1.2 데이터 변환

* 데이터 확인 - outlier, missing
* 데이터 변환
    * non-numeric: 'Sex', 'Embarked'
---
### M-1.2.1 outlier

* 데이터에 outlier가 있는지

```python
rdd.filter(lambda x:math.fabs(x-mean) < 3*stddev)
```
---
### M-1.2.2 Missing 데이터의 처리

* missing, not null이 있는지 확인

* agg()는 
    * aggregate함수 'avg', 'max', 'min', 'sum', 'count' 기능을 사용할 수 있다.
    * dict로 key, value
* 갯수를 세어보면, 'Age'와 'Fare'에 missing 값이 있다는 것을 알 수 있다.
---

```python
from pyspark.sql.functions import count
df.agg(*[count(c).alias(c) for c in df.columns]).show()

    +-----------+--------+------+----+----+----+-----+-----+------+----+-----+--------+-----------+
    |PassengerId|Survived|Pclass|Name| Sex| Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|testOrtrain|
    +-----------+--------+------+----+----+----+-----+-----+------+----+-----+--------+-----------+
    |       1309|    1309|  1309|1309|1309|1046| 1309| 1309|  1309|1308| 1309|    1309|       1309|
    +-----------+--------+------+----+----+----+-----+-----+------+----+-----+--------+-----------+
```
    
---


```python
def countNull(df,var):
    return df.where(df[var].isNull()).count()
missing = {c: countNull(df,c) for c in ['Survived','Age','SibSp','Parch','Fare']}

print missing

    {'Fare': 1, 'Age': 263, 'SibSp': 0, 'Survived': 0, 'Parch': 0}
```

---

```python
print df.filter("Age is null").show(5)
print df.filter("Fare is null").show(5)

    +-----------+--------+------+--------------------+------+----+-----+-----+------+------+-----+--------+-----------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|Ticket|  Fare|Cabin|Embarked|testOrtrain|
    +-----------+--------+------+--------------------+------+----+-----+-----+------+------+-----+--------+-----------+
    |          6|       0|     3|    Moran, Mr. James|  male|null|    0|    0|330877|8.4583|     |       Q|      train|
    |         18|       1|     2|Williams, Mr. Cha...|  male|null|    0|    0|244373|  13.0|     |       S|      train|
    |         20|       1|     3|Masselmani, Mrs. ...|female|null|    0|    0|  2649| 7.225|     |       C|      train|
    |         27|       0|     3|Emir, Mr. Farred ...|  male|null|    0|    0|  2631| 7.225|     |       C|      train|
    |         29|       1|     3|"O'Dwyer, Miss. E...|female|null|    0|    0|330959|7.8792|     |       Q|      train|
    +-----------+--------+------+--------------------+------+----+-----+-----+------+------+-----+--------+-----------+
    only showing top 5 rows
    
    +-----------+--------+------+------------------+----+----+-----+-----+------+----+-----+--------+-----------+
    |PassengerId|Survived|Pclass|              Name| Sex| Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|testOrtrain|
    +-----------+--------+------+------------------+----+----+-----+-----+------+----+-----+--------+-----------+
    |       1044|      99|     3|Storey, Mr. Thomas|male|60.5|    0|    0|  3701|null|     |       S|       test|
    +-----------+--------+------+------------------+----+----+-----+-----+------+----+-----+--------+-----------+
```
    
---

* 평균구하기
    * collect()의 결과는 Python List이므로, index '0', 컬럼명으로 평균값을 구할 수 있다.


```python
from pyspark.sql import functions as F
avgAge=df.agg(F.avg(df['Age']).alias('meanAge')).collect()
avgFare=df.agg(F.avg(df['Fare']).alias('meanFare')).collect()
print avgAge[0]['meanAge']
print avgFare[0]['meanFare']

    29.8811376673
    33.2954792813
```

---

```python
print df.groupBy().mean('Age').first()
print df.groupBy().mean('Fare').first()

    Row(avg(Age)=29.881137667304014)
    Row(avg(Fare)=33.29547928134553)
```

---

```python
df.describe(['Age']).show()

    +-------+------------------+
    |summary|               Age|
    +-------+------------------+
    |  count|              1046|
    |   mean|29.881137667304014|
    | stddev| 14.41349321127132|
    |    min|              0.17|
    |    max|              80.0|
    +-------+------------------+
```
    
---

* null 값의 처리
    * 
    pyspark.sql.functions.when(condition, value)
    * null이면 평균 값, 아니면 자신의 값을 유지한다.
    * not null -> nnDf
```
df4.na.fill({'age': 50, 'name': 'unknown'}).show()
```
---

```python
from pyspark.sql.functions import when,isnull
df=df.withColumn("Age", when(isnull(df['Age']), avgAge[0]['meanAge']).otherwise(df.Age))
df=df.withColumn("Fare", when(isnull(df['Fare']), avgFare[0]['meanFare']).otherwise(df.Fare))
#df.show(10)
```
---

```python
df.groupBy('testOrtrain').count().show()

    +-----------+-----+
    |testOrtrain|count|
    +-----------+-----+
    |      train|  891|
    |       test|  418|
    +-----------+-----+
```
    

---

```python
df.groupBy('Sex').count().show()

    +------+-----+
    |   Sex|count|
    +------+-----+
    |female|  466|
    |  male|  843|
    +------+-----+
```
    
---

* 이름으로 성별을 구별해 본다.
* 이름name에 학위 (Master, Dr.), 작위 등이 성별title 대신 사용된 경우가 있슴.


```python
import re
def getTitle(name):
    title=None
    if re.search(".*Mr\..*", name):
        title="male"
    elif re.search(".*[Miss|Mrs|Ms]\..*", name):
        title="female"
    return title

names=["Braund, Mr. Owen Harris",
       "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
       "Heikkinen, Miss. Laina",
       "Ms.hello",
       "No title"]
for n in names:
    print getTitle(n)

    male
    female
    female
    female
    None
```
---

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

getTitleUdf = udf(getTitle, StringType())

df = df.withColumn('Title', getTitleUdf(df['Name']))
 
df.select('testOrtrain','Name','Title','Sex')\
    .filter(df['testOrtrain']=='test')\
    .show(5,truncate=False)
df.groupBy('Title').count().show()
df.groupBy('Sex').count().show()

    +-----------+--------------------------------------------+------+------+
    |testOrtrain|Name                                        |Title |Sex   |
    +-----------+--------------------------------------------+------+------+
    |test       |Kelly, Mr. James                            |male  |male  |
    |test       |Wilkes, Mrs. James (Ellen Needs)            |female|female|
    |test       |Myles, Mr. Thomas Francis                   |male  |male  |
    |test       |Wirz, Mr. Albert                            |male  |male  |
    |test       |Hirvonen, Mrs. Alexander (Helga E Lindqvist)|female|female|
    +-----------+--------------------------------------------+------+------+
    only showing top 5 rows
    
    +------+-----+
    | Title|count|
    +------+-----+
    |  null|   19|
    |female|  533|
    |  male|  757|
    +------+-----+
    
    +------+-----+
    |   Sex|count|
    +------+-----+
    |female|  466|
    |  male|  843|
    +------+-----+
```
    
---


```python
df.printSchema()

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: string (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: string (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: string (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
     |-- testOrtrain: string (nullable = false)
```
    

---

```python
df=df.withColumn("SurvivedD",trainDf['Survived']\
    .cast("double"))\
    .drop('Survived')
```
---

```python
df.groupBy('SurvivedD').count().show()

    +---------+-----+
    |SurvivedD|count|
    +---------+-----+
    |      0.0|  549|
    |      1.0|  342|
    |     99.0|  418|
    +---------+-----+
```
---    



```python
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
#from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

SexIndexer = StringIndexer(inputCol="Sex", outputCol="SexI")
EmbarkedIndexer = StringIndexer(inputCol="Embarked", outputCol="EmbarkedI")
va = VectorAssembler(inputCols=["Pclass","SexI","Age","SibSp","Parch",\
                                "Fare","EmbarkedI"],\
                     outputCol="features")
pipeline = Pipeline(stages=[PclassIndexer,SexIndexer,ParchIndexer,\
                            FareIndexer,EmbarkedIndexer,va])
model = pipeline.fit(df)
myDf = model.transform(df)
```
---

```python
myDf.select('SurvivedD','features').show(10)

    +---------+--------------------+
    |SurvivedD|            features|
    +---------+--------------------+
    |      0.0|[3.0,0.0,22.0,1.0...|
    |      1.0|[1.0,1.0,38.0,1.0...|
    |      1.0|[3.0,1.0,26.0,0.0...|
    |      1.0|[1.0,1.0,35.0,1.0...|
    |      0.0|(7,[0,2,5],[3.0,3...|
    |      0.0|[3.0,0.0,29.88113...|
    |      0.0|(7,[0,2,5],[1.0,5...|
    |      0.0|[3.0,0.0,2.0,3.0,...|
    |      1.0|[3.0,1.0,27.0,0.0...|
    |      1.0|[2.0,1.0,14.0,1.0...|
    +---------+--------------------+
    only showing top 10 rows
```
    
---

* randomSplit()
Randomly splits this DataFrame with the provided weights.


```python
train=myDf.filter(myDf['testOrtrain']=='train')
testDf=myDf.filter(myDf['testOrtrain']=='test')
trainDf,validateDf = train.randomSplit([0.7,0.3],seed=11)

print "all num of rows: ",myDf.count()
print 'train num of rows: ',trainDf.count()
print 'validate num of rows: ',validateDf.count()
print 'test num of rows: ',testDf.count()

    all num of rows:  1309
    train num of rows:  628
    validate num of rows:  263
    test num of rows:  418
```
---

### M-1.3 모델링

#### LogisticRegression

이진 분류

```python
from pyspark.ml.classification import LogisticRegression
 
# regPara: lasso regularisation parameter (L1)
lr = LogisticRegression().\
    setLabelCol('SurvivedD').\
    setFeaturesCol('features').\
    setRegParam(0.0).\
    setMaxIter(100).\
    setElasticNetParam(0.)
lrModel=lr.fit(trainDf)
```
---

####  dt, rf

```python
dt = DecisionTreeClassifier(maxDepth = 3, labelCol ='index').fit(train)
rf = RandomForestClassifier(numTrees = 100, labelCol = 'index').fit(train)
```
---
### M-1.4 예측

### M-1.5 평가

* testDF를 만들어서
* 이진분류의 경우

구분 | 설명
-----|-----
rawPrediction | 이진분류 예측 또는 확률, double 또는 벡터
label | 실제 값

. The rawPrediction column can be of type double (binary 0/1 prediction, or probability of label 1) or of type vector (length-2 vector of raw predictions, scores, or label probabilities).
---

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator

lrDf = lrModel.transform(validateDf)
```
---

```python
lrDf.printSchema()

    root
     |-- PassengerId: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
     |-- testOrtrain: string (nullable = false)
     |-- Title: string (nullable = true)
     |-- SurvivedD: double (nullable = true)
     |-- PclassI: double (nullable = true)
     |-- SexI: double (nullable = true)
     |-- ParchI: double (nullable = true)
     |-- FareI: double (nullable = true)
     |-- EmbarkedI: double (nullable = true)
     |-- features: vector (nullable = true)
     |-- rawPrediction: vector (nullable = true)
     |-- probability: vector (nullable = true)
     |-- prediction: double (nullable = true)
    
```
---


```python
lrDf.select('SurvivedD','rawPrediction','probability','prediction').show()

    +---------+--------------------+--------------------+----------+
    |SurvivedD|       rawPrediction|         probability|prediction|
    +---------+--------------------+--------------------+----------+
    |      1.0|[-0.7209575821080...|[0.32718215168700...|       1.0|
    |      1.0|[-2.0503227895482...|[0.11401976925515...|       1.0|
    |      1.0|[-1.9554360708280...|[0.12396181876919...|       1.0|
    |      1.0|[-1.1467078201107...|[0.24109092679847...|       1.0|
    |      1.0|[-1.4698503211520...|[0.18696536548151...|       1.0|
    |      0.0|[1.69472254018706...|[0.84484421073581...|       0.0|
    |      1.0|[1.26032179504787...|[0.77908149803135...|       0.0|
    |      1.0|[-1.4501313540239...|[0.18998135132665...|       1.0|
    |      1.0|[-0.0619658876835...|[0.48451348315217...|       1.0|
    |      0.0|[2.56037173163627...|[0.92826721424349...|       0.0|
    |      1.0|[-0.8920008140330...|[0.29069710229074...|       1.0|
    |      0.0|[-1.2464028691324...|[0.22332344045916...|       1.0|
    |      0.0|[1.95032687007954...|[0.87548227929311...|       0.0|
    |      1.0|[-0.8497256926713...|[0.29949040291388...|       1.0|
    |      0.0|[-0.6189589760433...|[0.35001825308221...|       1.0|
    |      0.0|[1.73615921896138...|[0.85019855753063...|       0.0|
    |      0.0|[3.56369823671166...|[0.97244684148754...|       0.0|
    |      0.0|[0.86690971886304...|[0.70410226817396...|       0.0|
    |      1.0|[-1.5870055403151...|[0.16980561361907...|       1.0|
    |      1.0|[0.86052411729901...|[0.70277014583679...|       0.0|
    +---------+--------------------+--------------------+----------+
    only showing top 20 rows
```
    
---


```python
evaluator = BinaryClassificationEvaluator(rawPredictionCol = 'prediction',\
                                          labelCol='SurvivedD')
evaluator.evaluate(lrDf)




    0.7919513103962241

```

---
* ROC

```
val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTr ueCats)
   val nbPrCats = nbMetricsCats.areaUnderPR
   val nbRocCats = nbMetricsCats.areaUnderRO
```
---
* 개선
    * feature standardization

```    
val matrix = new RowMatrix(vectors)
val matrixSummary = matrix.computeColumnSummaryStatistics()
val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
```
---
## 문제 M-2: Kaggle Twitter US Airline Sentiment

* 원본 https://www.crowdflower.com/data-for-everyone/

* Google search - tweet sentiment corpus
    * http://help.sentiment140.com/home
    * http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/

* 압축을 풀면, 
     Tweets.csv와 database.sqlite 2 파일이 생성, 동일한 내용
---

* sqlite
```
$ sqlite3 data/kaggle/tweeterUSAirlineSentiment/database.sqlite 
SQLite version 3.11.0 2016-02-15 17:29:24
Enter ".help" for usage hints.
sqlite> .table
Tweets
```
---
* 14485 'negativereason_confidence' 제외한 건수

* ibm직원이 tweet을 변환해서 mlib한 거 https://github.com/castanan/w2v

구분 | 건수
-----|-----
데이터 행 | 14485
데이터 열 | 15 tweet_id, airline_sentiment, airline_sentiment_confidence, negativereason, negativereason_confidence, airline, airline_sentiment_gold, name, negativereason_gold, retweet_count, text, tweet_coord, tweet_created, tweet_location, user_timezone

---

1. 데이터 수집해서 dataframe.
* url, RT, punctuations, numbers, lowercase, emoticons
2. stop words
tokenize
tf-idf

---
### M-2.1 데이터 수집

#### Tweets.csv

* sentiment
    * positive, neutral, negative


```python
tDf = spark.read.format('com.databricks.spark.csv')\
    .options(header='true', inferschema='true')\
    .load('data/kaggle/tweeterUSAirlineSentiment/Tweets.csv')
tDf.take(1)

tDf.printSchema()
```

    root
     |-- tweet_id: string (nullable = true)
     |-- airline_sentiment: string (nullable = true)
     |-- airline_sentiment_confidence: string (nullable = true)
     |-- negativereason: string (nullable = true)
     |-- negativereason_confidence: string (nullable = true)
     |-- airline: string (nullable = true)
     |-- airline_sentiment_gold: string (nullable = true)
     |-- name: string (nullable = true)
     |-- negativereason_gold: string (nullable = true)
     |-- retweet_count: integer (nullable = true)
     |-- text: string (nullable = true)
     |-- tweet_coord: string (nullable = true)
     |-- tweet_created: string (nullable = true)
     |-- tweet_location: string (nullable = true)
     |-- user_timezone: string (nullable = true)
    



```python
tDf.select('negativereason_confidence').count()
```




    14650




```python
tDf.select('text', 'airline_sentiment_confidence',\
          'negativereason_confidence').show(10)
```

    +--------------------+----------------------------+-------------------------+
    |                text|airline_sentiment_confidence|negativereason_confidence|
    +--------------------+----------------------------+-------------------------+
    |@VirginAmerica Wh...|                         1.0|                         |
    |@VirginAmerica pl...|                      0.3486|                      0.0|
    |@VirginAmerica I ...|                      0.6837|                         |
    |"@VirginAmerica i...|                         1.0|                   0.7033|
    |@VirginAmerica an...|                         1.0|                      1.0|
    |@VirginAmerica se...|                         1.0|                   0.6842|
    |@VirginAmerica ye...|                      0.6745|                      0.0|
    |@VirginAmerica Re...|                       0.634|                         |
    |@virginamerica We...|                      0.6559|                         |
    |@VirginAmerica it...|                         1.0|                         |
    +--------------------+----------------------------+-------------------------+
    only showing top 10 rows
    


#### sqlite database.sqlite


```python
_df=spark.read.format('jdbc')\
    .options(
        url="jdbc:sqlite:./data/kaggle/tweeterUSAirlineSentiment/database.sqlite",
        dbtable="Tweets",
        driver="org.sqlite.JDBC"
    ).load()
```


```python
_df.printSchema()
```

    root
     |-- tweet_id: integer (nullable = false)
     |-- airline_sentiment: string (nullable = true)
     |-- airline_sentiment_confidence: decimal(38,18) (nullable = true)
     |-- negativereason: string (nullable = true)
     |-- negativereason_confidence: decimal(38,18) (nullable = true)
     |-- airline: string (nullable = true)
     |-- airline_sentiment_gold: string (nullable = true)
     |-- name: string (nullable = true)
     |-- negativereason_gold: string (nullable = true)
     |-- retweet_count: integer (nullable = true)
     |-- text: string (nullable = true)
     |-- tweet_coord: string (nullable = true)
     |-- tweet_created: string (nullable = true)
     |-- tweet_location: string (nullable = true)
     |-- user_timezone: string (nullable = true)
    


### M-2.2 데이터 변환

#### 탐색

* sentiment 구분
* csv에서 읽은 tDf와 차이가 있다.


```python
print tDf.groupBy('airline_sentiment').count().show()
print tDf.groupBy('airline')\
    .agg({'airline_sentiment': 'count'}).show()
```

    +--------------------+-----+
    |   airline_sentiment|count|
    +--------------------+-----+
    |            positive| 2363|
    |                null|    1|
    | this is where Ce...|    1|
    |             neutral| 3099|
    | this is where Ce...|    2|
    |            negative| 9178|
    |          [0.0, 0.0]|    1|
    | this is where Ce...|    1|
    | this is where Ce...|    1|
    |                    |    3|
    +--------------------+-----+
    
    None
    +--------------+------------------------+
    |       airline|count(airline_sentiment)|
    +--------------+------------------------+
    |         Delta|                    2222|
    |          null|                       4|
    |Virgin America|                     504|
    |        United|                    3822|
    |    US Airways|                    2913|
    |     Southwest|                    2420|
    |       Tijuana|                       5|
    |      American|                    2759|
    +--------------+------------------------+
    
    None



```python
print _df.groupBy('airline_sentiment').count().show()
print _df.groupBy('airline')\
    .agg({'airline_sentiment': 'count'}).show()
```

    +-----------------+-----+
    |airline_sentiment|count|
    +-----------------+-----+
    |         positive| 2334|
    |          neutral| 3069|
    |         negative| 9082|
    +-----------------+-----+
    
    None
    +--------------+------------------------+
    |       airline|count(airline_sentiment)|
    +--------------+------------------------+
    |         Delta|                    2222|
    |Virgin America|                     504|
    |        United|                    3822|
    |    US Airways|                    2913|
    |     Southwest|                    2420|
    |      American|                    2604|
    +--------------+------------------------+
    
    None



```python
import pyspark.sql.functions as F
total=_df.count()
sDf=(_df.groupBy('airline_sentiment').count()
    .withColumn('total',F.lit(total))
    .withColumn('fraction',F.expr('count/total')))
sDf.show()
```

    +-----------------+-----+-----+-------------------+
    |airline_sentiment|count|total|           fraction|
    +-----------------+-----+-----+-------------------+
    |         positive| 2334|14485| 0.1611322057300656|
    |          neutral| 3069|14485|0.21187435277873662|
    |         negative| 9082|14485| 0.6269934414911978|
    +-----------------+-----+-----+-------------------+
    


* 비율로 그림?


```python
#_tDf.cube("airline", _tDf.airline_sentiment).count()\
#    .orderBy("airline", "airline_sentiment").show()
_df.stat.crosstab("airline","airline_sentiment").show()

_df.groupBy('negativereason').count().show()
```

    +-------------------------+--------+-------+--------+
    |airline_airline_sentiment|negative|neutral|positive|
    +-------------------------+--------+-------+--------+
    |                    Delta|     955|    723|     544|
    |           Virgin America|     181|    171|     152|
    |               US Airways|    2263|    381|     269|
    |                   United|    2633|    697|     492|
    |                 American|    1864|    433|     307|
    |                Southwest|    1186|    664|     570|
    +-------------------------+--------+-------+--------+
    
    +--------------------+-----+
    |      negativereason|count|
    +--------------------+-----+
    |        Lost Luggage|  719|
    |           longlines|  177|
    |         Late Flight| 1650|
    |     Damaged Luggage|   73|
    |    Cancelled Flight|  829|
    |Customer Service ...| 2885|
    |Flight Attendant ...|  475|
    |                    | 5403|
    |          Bad Flight|  575|
    |          Can't Tell| 1176|
    |Flight Booking Pr...|  523|
    +--------------------+-----+
    


* 'tweet_location', 'retweet_count' 분석 - 비율, 지도위에??

* DateType()은 년월일 형식을 지원 "0001-01-01" through "9999-12-31".
* 'negativereason_confidence'
    * sqlite를 사용하면 BigDecimal오류가 
    'java.sql.SQLException: Bad value for type BigDecimal'
    * null 값이 많다.

변수명 | 값 | null
-----|-----|-----
airline_sentiment_confidence | 1~0의 값, 소수점 18자리까지 | null 값이 거의 없다.
negativereason_confidence | 상동 | null 값이 많다.


```python
from pyspark.sql.types import IntegerType, DateType, DoubleType, DecimalType, FloatType
_tDf=_df.withColumn("airline_sentiment_confidenceD",\
                    _df['airline_sentiment_confidence']\
                   .cast("double")).drop('airline_sentiment_confidence')
_tDf=_tDf.withColumn("negativereason_confidenceD",\
                     _tDf['negativereason_confidence']\
                   .cast("double")).drop('negativereason_confidence')
_tDf=_tDf.withColumn('tweet_createdDate', _tDf['tweet_created']\
                     .cast(DateType())).drop('tweet_created')
#_tDf=_tDf.withColumn('retweet_countI', _tDf['retweet_count']\
#                     .cast("integer")).drop('retweet_count')
_tDf=_tDf.drop('negativereason_confidenceD')
```


```python
from pyspark.sql.functions import count
_tDf.agg(*[count(c).alias(c) for c in _tDf.columns]).show()
```

    +--------+-----------------+--------------+-------+----------------------+-----+-------------------+-------------+-----+-----------+--------------+-------------+-----------------------------+-----------------+
    |tweet_id|airline_sentiment|negativereason|airline|airline_sentiment_gold| name|negativereason_gold|retweet_count| text|tweet_coord|tweet_location|user_timezone|airline_sentiment_confidenceD|tweet_createdDate|
    +--------+-----------------+--------------+-------+----------------------+-----+-------------------+-------------+-----+-----------+--------------+-------------+-----------------------------+-----------------+
    |   14485|            14485|         14485|  14485|                 14485|14485|              14485|        14485|14485|      14485|         14485|        14485|                        14485|            14485|
    +--------+-----------------+--------------+-------+----------------------+-----+-------------------+-------------+-----+-----------+--------------+-------------+-----------------------------+-----------------+
    


* 가려내기

패턴 | 설명
-----|-----
@[\w]+ | @로 시작하는 alphanumerics
[^\w] | alphanumeric이 아닌 한 글자, apostrophe, dot, etc.
\w+:\/\/\S+ | ://를 가지고 있는 url


```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import re

def myFilter(s):
    return ' '.join(re.sub("(@[\w]+)|([^\w])|(\w+:\/\/\S+)"," ",s).split())
myUdf = udf(myFilter, StringType())
filterDF = _tDf.withColumn("textFiltered", myUdf(_tDf['text']))
```


```python
from pyspark.ml.feature import *
re = RegexTokenizer(inputCol="textFiltered", outputCol="words", pattern="\\W")
wordsDf=re.transform(filterDF)
```


```python
wordsDf.select('text','words').take(3)
```




    [Row(text=u"@JetBlue's new CEO seeks the right balance to please passengers and Wall ... - Greenfield Daily Reporter http://t.co/LM3opxkxch", words=[u's', u'new', u'ceo', u'seeks', u'the', u'right', u'balance', u'to', u'please', u'passengers', u'and', u'wall', u'greenfield', u'daily', u'reporter']),
     Row(text=u'@JetBlue is REALLY getting on my nerves !! \U0001f621\U0001f621 #nothappy', words=[u'is', u'really', u'getting', u'on', u'my', u'nerves', u'nothappy']),
     Row(text=u'@united yes. We waited in line for almost an hour to do so. Some passengers just left not wanting to wait past 1am.', words=[u'yes', u'we', u'waited', u'in', u'line', u'for', u'almost', u'an', u'hour', u'to', u'do', u'so', u'some', u'passengers', u'just', u'left', u'not', u'wanting', u'to', u'wait', u'past', u'1am'])]




```python
from pyspark.ml.feature import StopWordsRemover
stop = StopWordsRemover(inputCol="words", outputCol="nostops")
```


```python
stopwords=list()

_stopwords=stop.getStopWords()
for e in _stopwords:
    stopwords.append(e)
_mystopwords=[u"나",u"너", u"우리"]
```


```python
for e in _mystopwords:
    stopwords.append(e)
print stopwords
```

    [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn', u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn', u'wasn', u'weren', u'won', u'wouldn', u'\ub098', u'\ub108', u'\uc6b0\ub9ac']



```python
stopDf=stop.transform(wordsDf)
stopDf.select('text','nostops').show()
```

    +--------------------+--------------------+
    |                text|             nostops|
    +--------------------+--------------------+
    |@JetBlue's new CE...|[new, ceo, seeks,...|
    |@JetBlue is REALL...|[really, getting,...|
    |@united yes. We w...|[yes, waited, lin...|
    |@united the we go...|[got, gate, iah, ...|
    |@SouthwestAir its...|[cool, bags, take...|
    |@united and don't...|[hope, nicer, fli...|
    |@united I like de...|[like, delays, le...|
    |@united, link to ...|[link, current, s...|
    |@SouthwestAir you...|[guys, hour, 2, p...|
    |@united I tried 2...|[tried, 2, dm, wo...|
    |Wanted to get my ...|[wanted, get, bag...|
    |@united please se...|[please, see, fli...|
    |@united still wai...|[still, waiting, ...|
    |@united even thou...|[even, though, te...|
    |@USAirways how's ...|[us, 1797, lookin...|
    |@SouthwestAir nic...|[nice, work, update]|
    |@united i have it...|[items, sentiment...|
    |@SouthwestAir We ...|[stuck, sju, seve...|
    |@JetBlue CEO weig...|[ceo, weighs, pro...|
    |@USAirways @Ameri...|[r, u, supposed, ...|
    +--------------------+--------------------+
    only showing top 20 rows
    



```python
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol="words", outputCol="cv", vocabSize=30,minDF=1.0)
cvModel = cv.fit(wordsDf)
cvDf = cvModel.transform(wordsDf)

cvDf.collect()
cvDf.select('text','words','cv').show()
```

    +--------------------+--------------------+--------------------+
    |                text|               words|                  cv|
    +--------------------+--------------------+--------------------+
    |@JetBlue's new CE...|[s, new, ceo, see...|(30,[0,2,8,22],[1...|
    |@JetBlue is REALL...|[is, really, gett...|(30,[7,9,10],[1.0...|
    |@united yes. We w...|[yes, we, waited,...|(30,[0,5,11,21,28...|
    |@united the we go...|[the, we, got, in...|(30,[0,2,4,6,7,8,...|
    |@SouthwestAir its...|[its, cool, that,...|(30,[2,3,7,9,12,1...|
    |@united and don't...|[and, don, t, hop...|(30,[0,3,4,5,6,8,...|
    |@united I like de...|[i, like, delays,...|(30,[1,2,3,4,7,15...|
    |@united, link to ...|[link, to, curren...|(30,[0,13,14,18,2...|
    |@SouthwestAir you...|[you, guys, there...|(30,[4,7,14,24,28...|
    |@united I tried 2...|[i, tried, 2, dm,...|(30,[1,12,21],[1....|
    |Wanted to get my ...|[wanted, to, get,...|(30,[0,3,7,9,10,2...|
    |@united please se...|[please, see, a, ...|(30,[0,1,3,5,6,7,...|
    |@united still wai...|[still, waiting, ...|(30,[3,5],[1.0,1.0])|
    |@united even thou...|[even, though, te...|      (30,[1],[2.0])|
    |@USAirways how's ...|[how, s, us, 1797...|     (30,[22],[1.0])|
    |@SouthwestAir nic...|[nice, work, on, ...|(30,[2,7],[1.0,1.0])|
    |@united i have it...|[i, have, items, ...|(30,[1,14,17,19],...|
    |@SouthwestAir We ...|[we, have, been, ...|(30,[0,5,8,10,11,...|
    |@JetBlue CEO weig...|[ceo, weighs, pro...|          (30,[],[])|
    |@USAirways @Ameri...|[how, r, u, suppo...|(30,[0,13,18,27],...|
    +--------------------+--------------------+--------------------+
    only showing top 20 rows
    


### M-2.3 모델링

#### Logistic

* multiclass



## 문제 M-3: Spark MLib Decision Tree

* 참조 https://www.codementor.io/spark/tutorial/spark-python-mllib-decision-trees

* 1단계: 데이터 수집
* 2단계: 데이터 준비
* 3단계: 모델링
* 4단계: 예측
* 5단계: 평가

### 1단계: 데이터 수집

http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
This is the data set used for The Third International Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with KDD-99 The Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network intrusion detector, a predictive model capable of distinguishing between ``bad'' connections, called intrusions or attacks, and ``good'' normal connections. This database contains a standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment.


* url에서 데이터를 내려받는다.
* data 디렉토리에 저장한다.
* 데이터를 내려 받아 놓았다면, 반복하지 않고 있는 파일을 읽는다.

* train data


```python
import os
import urllib

_url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz'
_trainFn=os.path.join(os.getcwd(),'data','kddcup.data.gz')
if(not os.path.exists(_trainFn)):
    print "%s data does not exist! retrieving.." % _trainFn
    _trainFn=urllib.urlretrieve(_url,_trainFn)

```


```python
_trainRdd = spark.sparkContext.textFile(_trainFn)
print _trainRdd.count()
```

    4898431



```python
_trainRdd.take(1)
```




    [u'0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.']



* test data


```python
_url2 = 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz'
_testFn=os.path.join(os.getcwd(),'data','corrected.gz')
if(not os.path.exists(_testFn)):
    print "%s data does not exist! retrieving.." % _testFn
    _testFn=urllib.urlretrieve(_url,_testFn)

```


```python
_testRdd = spark.sparkContext.textFile(_testFn)
print _testRdd.count()
```

    311029



```python
_testRdd.take(1)
```




    [u'0,udp,private,SF,105,146,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,254,1.00,0.01,0.00,0.00,0.00,0.00,0.00,0.00,normal.']



### 2단계: 데이터 준비

* 2-1 csv를 분리한다.
* 2-2 변수를 확인하여, 연속 값 또는 명목 값을 가지도록 한다.
    * 알파벳은 명목척도로 변환한다.
* 2-3 train data를 생성한다.
    * features 41개 - protocols, services, flags
    * label - 마지막 42번째 열 (attack = 0 if 'normal.', else 1)

변수명 | protocls | services | flags | ... | attack
-----|-----|-----|-----|-----|-----
인덱스 | 1 | 2 | 3| ... | 42
데이터 값 예 | tcp | http | SF | ... | normal


* 2-1 csv를 분리한다.
    * csv를 컴마로 분리하여, 2차원 데이터로 구조화한다.


```python
_train = _trainRdd.map(lambda x: x.split(","))
_test = _testRdd.map(lambda x: x.split(","))
```


```python
print len(_train.first()), _train.first()
print len(_test.first()), _test.first()
```

    42 [u'0', u'tcp', u'http', u'SF', u'215', u'45076', u'0', u'0', u'0', u'0', u'0', u'1', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'1', u'1', u'0.00', u'0.00', u'0.00', u'0.00', u'1.00', u'0.00', u'0.00', u'0', u'0', u'0.00', u'0.00', u'0.00', u'0.00', u'0.00', u'0.00', u'0.00', u'0.00', u'normal.']
    42 [u'0', u'udp', u'private', u'SF', u'105', u'146', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'1', u'1', u'0.00', u'0.00', u'0.00', u'0.00', u'1.00', u'0.00', u'0.00', u'255', u'254', u'1.00', u'0.01', u'0.00', u'0.00', u'0.00', u'0.00', u'0.00', u'0.00', u'normal.']


* 2-2 변수를 확인하여, 연속 값 또는 명목 값을 가지도록 한다.
    * 2,3,4번째 속성에 알파벳이 있다. 명목 값을 구하기 위해, 중복 값을 제외하고 key를 구한다.


```python
protocols = _train.map(lambda x: x[1]).distinct().collect()
services = _train.map(lambda x: x[2]).distinct().collect()
flags = _train.map(lambda x: x[3]).distinct().collect()
```


```python
print len(protocols), protocols
print len(services), services
print len(flags), flags
```

    3 [u'udp', u'icmp', u'tcp']
    70 [u'urp_i', u'http_443', u'Z39_50', u'smtp', u'domain', u'private', u'echo', u'time', u'shell', u'red_i', u'eco_i', u'sunrpc', u'ftp_data', u'urh_i', u'pm_dump', u'pop_3', u'pop_2', u'systat', u'ftp', u'uucp', u'whois', u'harvest', u'netbios_dgm', u'efs', u'remote_job', u'daytime', u'ntp_u', u'finger', u'ldap', u'netbios_ns', u'kshell', u'iso_tsap', u'ecr_i', u'nntp', u'http_2784', u'printer', u'domain_u', u'uucp_path', u'courier', u'exec', u'aol', u'netstat', u'telnet', u'gopher', u'rje', u'sql_net', u'link', u'ssh', u'netbios_ssn', u'csnet_ns', u'X11', u'IRC', u'tftp_u', u'login', u'supdup', u'name', u'nnsp', u'mtp', u'http', u'bgp', u'ctf', u'hostnames', u'klogin', u'vmnet', u'tim_i', u'discard', u'imap4', u'auth', u'other', u'http_8001']
    11 [u'OTH', u'RSTR', u'S3', u'S2', u'S1', u'S0', u'RSTOS0', u'REJ', u'SH', u'RSTO', u'SF']


* 2-3 train data를 생성한다.
    * LabeledPoint 형식으로 만든다.
    * feature 생성 - 명목척도로 만든다.
        * protocols는 알파벳, 이를 key를 사용하여 명목척도로 만든다.
        * services는 알파벳, 이를 key를 사용하여 명목척도로 만든다.
        * flags는 알파벳, 이를 key를 사용하여 명목척도로 만든다.
        * features는 numpy array를 사용하거나, Python list를 사용한다.
    * class 생성
        * 'normal.'이면 0
        * 아니면 1

* protocols에 대한 데이터 생성 해보기
    * 데이터 항목이 키 값에 있으면, 키 값을 넣는다 (index()).    
    * train data에서 key를 생성했기 때문에, test data에 없을 수 있다. 이 경우 최대 값을 넣는다 (len(), 임의의 값을 넣어도 좋다)

* index()는 list의 index를 알려 준다. 


```python
protocols.index('tcp')
```




    2



* 1건에 대해 LabeledPoint 생성 해보기


```python
from pyspark.mllib.regression import LabeledPoint
line=[u'0', u'tcp', u'http', u'SF', u'215', u'45076', u'0', u'0', u'0', u'0',\
      u'0', u'1', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0', u'0',\
      u'1', u'1', u'0.00', u'0.00', u'0.00', u'0.00', u'1.00', u'0.00', u'0.00',\
      u'0', u'0', u'0.00', u'0.00', u'0.00', u'0.00', u'0.00', u'0.00', u'0.00',\
      u'0.00', u'normal.']

feature=line[0:-1]
feature[1] = protocols.index(line[1]) if line[1] in protocols else len(protocols)
feature[2] = services.index(line[2]) if line[2] in services else len(services)
feature[3] = flags.index(line[3]) if line[3] in flags else len(flags)
attack = 0.0 if line[-1]=='normal.' else 1.0
LabeledPoint(attack, [float(x) for x in feature])
```




    LabeledPoint(0.0, [0.0,2.0,58.0,10.0,215.0,45076.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])




```python
for i,e in enumerate(line):
    print i,e
```

    0 0
    1 tcp
    2 http
    3 SF
    4 215
    5 45076
    6 0
    7 0
    8 0
    9 0
    10 0
    11 1
    12 0
    13 0
    14 0
    15 0
    16 0
    17 0
    18 0
    19 0
    20 0
    21 0
    22 1
    23 1
    24 0.00
    25 0.00
    26 0.00
    27 0.00
    28 1.00
    29 0.00
    30 0.00
    31 0
    32 0
    33 0.00
    34 0.00
    35 0.00
    36 0.00
    37 0.00
    38 0.00
    39 0.00
    40 0.00
    41 normal.


* LabeledPoint를 생성한다.


```python
from pyspark.mllib.regression import LabeledPoint
import numpy as np

def createLP(line):
    features=line[0:-1]
    features[1] = protocols.index(line[1]) if line[1] in protocols else len(protocols)
    features[2] = services.index(line[2]) if line[2] in services else len(services)
    features[3] = flags.index(line[3]) if line[3] in flags else len(flags)
    attack = 0.0 if line[-1]=='normal.' else 1.0
    lp=LabeledPoint(attack, [float(x) for x in features])
    return lp

trainRdd = _train.map(createLP)
testRdd = _test.map(createLP)
```


```python
print trainRdd.first()
print testRdd.first()
```

    (0.0,[0.0,2.0,58.0,10.0,215.0,45076.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    (0.0,[0.0,0.0,5.0,10.0,105.0,146.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,255.0,254.0,1.0,0.01,0.0,0.0,0.0,0.0,0.0,0.0])



```python
trainRdd.count()
```




    4898431




```python
testRdd.count()
```




    311029



### 3단계: 모델링

* 입력변수를 정의한다.

입력변수 | 설명
-------|-------
data | RDD of LabeledPoint
numClasses | 분류 class 수
categoricalFeaturesInfo | 명목척도의 Map (연속변수는 Map에 넣지 않음)
impurity| "entropy" 또는 "gini"
maxDepth | 트리의 최대 깊이 0 means 1 leaf node. Depth 1 means 1 internal node + 2 leaf nodes.
maxBins| Number of bins used for finding splits at each node.



```python
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
treeModel = DecisionTree.trainClassifier(trainRdd, numClasses=2, 
              categoricalFeaturesInfo={1: len(protocols), 2: len(services), 3: len(flags)},
              impurity='gini', maxDepth=4, maxBins=100)
```

### 4단계: 예측


```python
predictions = treeModel.predict(testRdd.map(lambda p: p.features))
labels_and_preds = testRdd.map(lambda p: p.label).zip(predictions)
```

### 5단계: 평가


```python
test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(testRdd.count())
```


```python
print test_accuracy
```

    0.918795353488



```python
print treeModel.toDebugString()
```

    DecisionTreeModel classifier of depth 4 with 29 nodes
      If (feature 22 <= 55.0)
       If (feature 3 in {2.0,3.0,4.0,7.0,9.0,10.0})
        If (feature 2 in {0.0,3.0,5.0,7.0,8.0,9.0,12.0,13.0,15.0,18.0,26.0,27.0,32.0,36.0,42.0,50.0,51.0,52.0,58.0,64.0,67.0,68.0})
         If (feature 34 <= 0.91)
          Predict: 0.0
         Else (feature 34 > 0.91)
          Predict: 1.0
        Else (feature 2 not in {0.0,3.0,5.0,7.0,8.0,9.0,12.0,13.0,15.0,18.0,26.0,27.0,32.0,36.0,42.0,50.0,51.0,52.0,58.0,64.0,67.0,68.0})
         If (feature 4 <= 22.0)
          Predict: 1.0
         Else (feature 4 > 22.0)
          Predict: 0.0
       Else (feature 3 not in {2.0,3.0,4.0,7.0,9.0,10.0})
        If (feature 33 <= 0.3)
         If (feature 5 <= 0.0)
          Predict: 1.0
         Else (feature 5 > 0.0)
          Predict: 0.0
        Else (feature 33 > 0.3)
         If (feature 22 <= 2.0)
          Predict: 0.0
         Else (feature 22 > 2.0)
          Predict: 1.0
      Else (feature 22 > 55.0)
       If (feature 5 <= 0.0)
        If (feature 11 <= 0.0)
         If (feature 2 in {0.0})
          Predict: 0.0
         Else (feature 2 not in {0.0})
          Predict: 1.0
        Else (feature 11 > 0.0)
         If (feature 2 in {12.0})
          Predict: 0.0
         Else (feature 2 not in {12.0})
          Predict: 1.0
       Else (feature 5 > 0.0)
        If (feature 29 <= 0.08)
         If (feature 2 in {3.0,4.0,26.0,36.0,42.0,58.0,68.0})
          Predict: 0.0
         Else (feature 2 not in {3.0,4.0,26.0,36.0,42.0,58.0,68.0})
          Predict: 1.0
        Else (feature 29 > 0.08)
         Predict: 1.0
    


* chi
https://www.codementor.io/jadianes/spark-mllib-logistic-regression-du107neto
```
def parse_interaction_categorical(line):
    line_split = line.split(",")
    clean_line_split = line_split[6:41]
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0
    return LabeledPoint(attack, 
        array([float(x) for x in clean_line_split]))

training_data_categorical = raw_data.map(parse_interaction_categorical)
```

## 문제 S-4: Spark MLib LDA

https://www.kaggle.com/c/crowdflower-search-relevance

https://github.com/hacertilbec/LDA-spark-python/blob/master/SparkLDA.py#L6

### newsgroup

* 문제: computer 관련 분류, 토픽모델링
* 'mini_newsgroups.tar.gz' newsgroup을 내려받기
* 내려받고 압축을 푼다 'tar -xvfz mini_newsgroups.tar.gz'
* 디렉토리를 textFile() 인자로 넣으면, 구성 파일의 내용을 읽는다.

* original 20 Newsgroups dataset에서 무작위로 선별하여 mini 20 Newsgroups dataset
* 100 articles from the following 20 Usenet newsgroups:

```
alt.atheism
comp.graphics
comp.os.ms-windows.misc
comp.sys.ibm.pc.hardware
comp.sys.mac.hardware
comp.windows.x
misc.forsale
rec.autos
rec.motorcycles
rec.sport.baseball
rec.sport.hockey
sci.crypt
sci.electronics
sci.med
sci.space
soc.religion.christian
talk.politics.guns
talk.politics.mideast
talk.politics.misc
talk.religion.misc
```

The articles from each of the 20 Newsgroups are arranged by topic in filesystem directories
– 20 directories, one per topic
– 100 files in each directory, one file = one document

* sklearn의 datasets


```python
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

list(newsgroups_train.target_names)
```




    ['alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc']



* archive.ics.uci.edu에서 내려 받기


```python
import os
import urllib
_url="https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz"
_trainFn=os.path.join(os.getcwd(),'data','mini_newsgroups.tar.gz')
if(not os.path.exists(_trainFn)):
    print "%s data does not exist! retrieving.." % _trainFn
    _trainFn=urllib.urlretrieve(_url,_trainFn)
```

    /home/jsl/Code/git/bb/jsl/pyds/data/mini_newsgroups.tar.gz data does not exist! retrieving..


* wildcards '*'를 사용하면 하위 디렉토리 모든 파일
* file


```python
import os
newsDir=os.path.join("data","mini_newsgroups","*")
_testRdd = spark.sparkContext.wholeTextFiles(newsDir)
print _testRdd.count()
```

    2000



```python
_file=_testRdd.first()[0]
_text=_testRdd.first()[1]
```

* \n\n 다음에 본문이 시작된다. 본문을 꺼내려면 \n\n을 찾는다.


```python
print _file
_file.split('/')[-2]
```

    file:/home/jsl/Code/git/bb/jsl/pyds/data/mini_newsgroups/sci.space/61262





    u'sci.space'




```python
print _text
```

    Xref: cantaloupe.srv.cs.cmu.edu sci.space:61262 sci.astro:35052
    Path: cantaloupe.srv.cs.cmu.edu!magnesium.club.cc.cmu.edu!news.sei.cmu.edu!cis.ohio-state.edu!pacific.mps.ohio-state.edu!zaphod.mps.ohio-state.edu!darwin.sura.net!bogus.sura.net!news-feed-1.peachnet.edu!umn.edu!msus1.msus.edu!vax1.mankato.msus.edu!belgarath
    Newsgroups: sci.space,sci.astro
    Subject: Re: Gamma Ray Bursters. Where are they?
    Message-ID: <1993Apr26.192535.1@vax1.mankato.msus.edu>
    From: belgarath@vax1.mankato.msus.edu
    Date: 26 Apr 93 19:25:35 -0600
    References: <1radsr$att@access.digex.net> <1993Apr24.221344.1@vax1.mankato.msus.edu> 
     <93116.093828SAUNDRSG@QUCDN.QueensU.CA> <1993Apr26.141114.19777@midway.uchicago.edu>
    Organization: Mankato State University
    Nntp-Posting-Host: vax1.mankato.msus.edu
    Lines: 53
    
    In article <1993Apr26.141114.19777@midway.uchicago.edu>, pef1@quads.uchicago.edu (it's enrico palazzo!) writes:
    >> = From: Graydon <SAUNDRSG@QUCDN.QueensU.CA>
    > 
    >> If all of these things have been detected in space, has anyone
    >> looked into possible problems with the detectors?
    > 
    >> That is, is there some mechanism (cosmic rays, whatever) that
    >> could cause the dector to _think_ it was seeing one of these
    >> things?
    > 
    >> Graydon
    > 
    > That would not explain why widely separated detectors, such as on Ulysses
    > and PVO and Ginga et al., would see a burst at the same time(*).  In fact, be-
    > fore BATSE, having this widely separated "Interplanetary Network" was the
    > only sure way to locate a random burst.  With only one detector, one cannot
    > locate a burst (except to say "It's somewhere in the field of view.").  With
    > two detectors, one can use the time that the burst is seen in each detector
    > to narrow the location to a thin annulus on the sky.  With three detectors,
    > one gets intersecting annuli, giving two possible locations.  If one of these
    > locations is impossible (because, say, the Earth blocked that part of the 
    > sky), voila, you have an error box.
    > 
    > BATSE, by having 8 detectors of its own, can do its own location determination,
    > but only to within about 3 degrees (would someone at GSFC, like David, like
    > to comment on the current state of location determination?).  Having inde-
    > pendent sightings by other detectors helps drive down the uncertainty.
    > 
    > You did touch on something that you didn't mean to, though.  Some believe
    > (in a reference that I have somewhere) that absorption-like features seen
    > in a fraction of GRBs can actually be caused by the detector.  It would be
    > a mean, nasty God, though, that would have a NaI crystal act like a 10^12 Gauss
    > neutron star...but this is getting too far afield.
    > 
    > Peter
    > peterf@oddjob.uchicago.edu
    > 
    
            All of this is VERY valid and very true.  But to add to this
    explaniation, each individual detector also has a built in fail-safe, just so
    the detector does not read the background radiation(i.e. cosmic rays), 
    if I remember right, the detectors go off about 3 to 5 sigma above the 
    background.  This is so they don't catch particularly energetic cosmic rays
    that would normally set it off. Even with this buffer, they still have to throw
    out something like 1/2 of the bursts that they DO get, because of the Earth's
    Van Allen Belts, the South Atlantic Anomaly, the Sun,  if I remember right,
    there is either a radar station, or a radio station in Australia, and there are
    a couple other sources as well.  
                                                    -jeremy
                                                    belgarath@vax1.mankato.msus.edu
    
    
    
    


* DataFrame을 생성한다.
    * 문서번호를 생성하기 위해 zipWithIndex()를 사용한다.
    * zipWithIndex()는 각 요소에 id를 0부터 순서대로 생성해서 묶는다.


```python
spark.sparkContext.parallelize(["x", "y", "z"]).zipWithIndex().collect()
```




    [('x', 0), ('y', 1), ('z', 2)]




```python
from pyspark.sql import Row
path = 'data/ds_spark_wiki.txt'
_rdd = spark.sparkContext.textFile(path).zipWithIndex()\
    .map(lambda(words,idd): Row(idd= idd, words = words.split(" ")))
df = spark.createDataFrame(_rdd)

df.printSchema()
```

    root
     |-- idd: long (nullable = true)
     |-- words: array (nullable = true)
     |    |-- element: string (containsNull = true)
    



```python
_rdd=_testRdd.map(lambda x:x[1])
_rdd=_rdd.zipWithIndex()
df=_rdd.toDF(["text","id"])
```


```python
df.show(3)
```

    +--------------------+---+
    |                text| id|
    +--------------------+---+
    |Xref: cantaloupe....|  0|
    |Xref: cantaloupe....|  1|
    |Newsgroups: sci.s...|  2|
    +--------------------+---+
    only showing top 3 rows
    



```python
* filter

* regextokenizer
* stopwords
* countvectorizer
* lda
```

* filter

패턴 | 설명
-----|-----
@[\w]+ | @로 시작하는 alphanumerics
[^\w] | , . > ! : - 등 alphanumeric이 아닌 한 글자
\w+:\/\/\S+ | ://를 가지고 있는 url
\d+ | 숫자로만 되어 있는 경우




```python
mystr="""Xref: cantaloupe.srv.cs.cmu.edu sci.space:61262 sci.astro:35052
Path: cantaloupe.srv.cs.cmu.edu!magnesium.club.cc.cmu.edu!news.sei.cmu.edu!cis.ohio-state.edu!pacific.mps.ohio-state.edu!zaphod.mps.ohio-state.edu!darwin.sura.net!bogus.sura.net!news-feed-1.peachnet.edu!umn.edu!msus1.msus.edu!vax1.mankato.msus.edu!belgarath
Newsgroups: sci.space,sci.astro
Subject: Re: Gamma Ray Bursters. Where are they?
Message-ID: <1993Apr26.192535.1@vax1.mankato.msus.edu>
From: belgarath@vax1.mankato.msus.edu
Date: 26 Apr 93 19:25:35 -0600
References: <1radsr$att@access.digex.net> <1993Apr24.221344.1@vax1.mankato.msus.edu> 
 <93116.093828SAUNDRSG@QUCDN.QueensU.CA> <1993Apr26.141114.19777@midway.uchicago.edu>
Organization: Mankato State University
Nntp-Posting-Host: vax1.mankato.msus.edu
Lines: 53

In article <1993Apr26.141114.19777@midway.uchicago.edu>, pef1@quads.uchicago.edu (it's enrico palazzo!) writes:
>> = From: Graydon <SAUNDRSG@QUCDN.QueensU.CA>"""
```


```python
import re
p=re.compile("(@[\w]+)|([^\w])|(\w+:\/\/\S+)|([^0-9]*)")
res=filter(p.search,mystr.split())
for e in res:
    print e
```

    Xref:
    cantaloupe.srv.cs.cmu.edu
    sci.space:61262
    sci.astro:35052
    Path:
    cantaloupe.srv.cs.cmu.edu!magnesium.club.cc.cmu.edu!news.sei.cmu.edu!cis.ohio-state.edu!pacific.mps.ohio-state.edu!zaphod.mps.ohio-state.edu!darwin.sura.net!bogus.sura.net!news-feed-1.peachnet.edu!umn.edu!msus1.msus.edu!vax1.mankato.msus.edu!belgarath
    Newsgroups:
    sci.space,sci.astro
    Subject:
    Re:
    Gamma
    Ray
    Bursters.
    Where
    are
    they?
    Message-ID:
    <1993Apr26.192535.1@vax1.mankato.msus.edu>
    From:
    belgarath@vax1.mankato.msus.edu
    Date:
    26
    Apr
    93
    19:25:35
    -0600
    References:
    <1radsr$att@access.digex.net>
    <1993Apr24.221344.1@vax1.mankato.msus.edu>
    <93116.093828SAUNDRSG@QUCDN.QueensU.CA>
    <1993Apr26.141114.19777@midway.uchicago.edu>
    Organization:
    Mankato
    State
    University
    Nntp-Posting-Host:
    vax1.mankato.msus.edu
    Lines:
    53
    In
    article
    <1993Apr26.141114.19777@midway.uchicago.edu>,
    pef1@quads.uchicago.edu
    (it's
    enrico
    palazzo!)
    writes:
    >>
    =
    From:
    Graydon
    <SAUNDRSG@QUCDN.QueensU.CA>



```python

```


```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import re

def myFilter(s):
    return ' '.join(re.sub("(@[\w]+)|([^\w])|(\w+:\/\/\S+)"," ",s).split())
myUdf = udf(myFilter, StringType())
filterDF = _tDf.withColumn("textFiltered", myUdf(_tDf['text']))
```


```python
from pyspark.ml.feature import *
re = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
wordsDf=re.transform(df)
```


```python
from pyspark.ml.feature import StopWordsRemover
stop = StopWordsRemover(inputCol="words", outputCol="nostops")
stopDf=stop.transform(wordsDf)
```


```python
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol="nostops", outputCol="vectors")
cvModel = cv.fit(stopDf)
cvDf = cvModel.transform(stopDf)

cvDf.printSchema()

cvDf.count()
```

    root
     |-- text: string (nullable = true)
     |-- id: long (nullable = true)
     |-- words: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- nostops: array (nullable = true)
     |    |-- element: string (containsNull = true)
     |-- vectors: vector (nullable = true)
    





    2000




```python
voca=cvModel.vocabulary
nword=len(voca)
ntopic=10
ndoc=df.count()
print "단어수: ",nword
print "문서수: ",ndoc
for i,v in enumerate(voca):
    if(i%500==True):
        print i,v,'-'
```

    단어수:  50532
    문서수:  2000
    1 cmu -
    501 current -
    1001 picture -
    1501 athena -
    2001 aurora -
    2501 chi -
    3001 encrypted -
    3501 lock -
    4001 depth -
    4501 intend -
    5001 headed -
    5501 judging -
    6001 initially -
    6501 highway -
    7001 768 -
    7501 wondered -
    8001 wickman -
    8501 polite -
    9001 mathematica -
    9501 allies -
    10001 upgrades -
    10501 persecuted -
    11001 hype -
    11501 infallibility -
    12001 699 -
    12501 loyd -
    13001 ubvmsd -
    13501 venky -
    14001 ddbeezer -
    14501 cci -
    15001 rpao -
    15501 meltdown -
    16001 homayoon -
    16501 hereditary -
    17001 15806 -
    17501 nred -
    18001 ludd -
    18501 feirerra -
    19001 n1nzu -
    19501 dergisi -
    20001 flashlight -
    20501 446 -
    21001 fnord -
    21501 inverting -
    22001 burge -
    22501 8994 -
    23001 retreating -
    23501 bergqvist -
    24001 cooked -
    24501 cursive -
    25001 kaiser -
    25501 havens -
    26001 mannikin -
    26501 entrangeres -
    27001 neccessity -
    27501 q2g -
    28001 93apr23102102 -
    28501 m0108ll -
    29001 newcott -
    29501 7754 -
    30001 6gre -
    30501 _biological -
    31001 brewers -
    31501 1993apr03 -
    32001 cu6 -
    32501 roland_behunin -
    33001 comprimised -
    33501 jrsc5frfr -
    34001 7rb -
    34501 colum -
    35001 rbnmtm -
    35501 _lots_ -
    36001 tenses -
    36501 15970 -
    37001 146ij_7ej_6k -
    37501 chattels -
    38001 fj1200 -
    38501 c5tb2f -
    39001 57khz -
    39501 rlkyrm8 -
    40001 031823 -
    40501 v95pjex -
    41001 140723 -
    41501 336683788851850579e -
    42001 epistimology -
    42501 592586 -
    43001 awdpa -
    43501 atrophy -
    44001 martyr -
    44501 kibbitzer -
    45001 hypothically -
    45501 a84ga84g -
    46001 terrestrial -
    46501 tatoo -
    47001 3mo -
    47501 haley -
    48001 wonderland_ -
    48501 n0m -
    49001 petr -
    49501 eyeballs -
    50001 burrito -
    50501 ramblings -


* y는 ml Vectors이다. mllib에서 사용하려면 변환이 필요하다.


```python
from pyspark.mllib.linalg import Vectors
# ndoc -> corpus_size = cvDf.count()  # total number of words
testRdd = cvDf.select("id", "vectors")\
    .rdd.map(lambda (x,y): [x,Vectors.fromML(y)]).cache()
```


```python
from pyspark.mllib.clustering import LDA, LDAModel
ldaModel = LDA.train(testRdd, k=ntopic,maxIterations=100,optimizer='online')

topic=ldaModel.describeTopics(maxTermsPerTopic=10)

for k in range(ntopic):
    print "* Topic: ", k, topic[k]

for k in range(ntopic):
    print "Topic ",k
    for w in topic[k][0]:
        print voca[w],
    print "\n-----"
```

    * Topic:  0 ([1218, 2555, 3099, 3673, 1305, 328, 967, 1453, 3807, 4681], [0.00446606182130309, 0.0033798280978327177, 0.003165296452491259, 0.0029992725381732706, 0.0027726988486314846, 0.0027442033008861708, 0.002675320971691053, 0.002426857581933155, 0.002155048128705299, 0.0020344004763281297])
    * Topic:  1 ([1686, 2374, 2203, 1542, 4371, 1343, 4202, 780, 5294, 1979], [0.004612963520161311, 0.003638496210463725, 0.0033858331735928146, 0.0026107819792790064, 0.0020867475502417358, 0.0020596069061911766, 0.0019152543492338156, 0.0017067101731930736, 0.0013959371145195812, 0.0013867822229835089])
    * Topic:  2 ([2618, 5548, 3009, 3609, 2778, 4239, 4592, 4839, 3477, 4617], [0.0034520127948244202, 0.0024376583265894084, 0.002397474729936783, 0.0022175644371884426, 0.002141850348482497, 0.001987419729703522, 0.0017942425559203986, 0.0016703075245876108, 0.0015030039514219537, 0.0014750798620378688])
    * Topic:  3 ([39, 144, 107, 66, 239, 134, 353, 150, 225, 94], [0.03245218526829409, 0.021224591395944726, 0.015368353457465607, 0.015060774816669915, 0.013232425128893055, 0.012047438707484674, 0.008631414662419045, 0.0070337560298776615, 0.006991642472555907, 0.006682438190530347])
    * Topic:  4 ([659, 153, 88, 1785, 2408, 575, 1294, 1653, 2392, 2841], [0.007836003301457228, 0.006191641423758272, 0.004294832380715145, 0.0031002673838980118, 0.0030378929148773596, 0.0030009507107107082, 0.00298262834277637, 0.0025512929111534176, 0.001904133482756802, 0.0018487997125403418])
    * Topic:  5 ([285, 2566, 1001, 2057, 2471, 4421, 3475, 5269, 9733, 5889], [0.0036464412040949326, 0.003618818080428977, 0.003563699910994846, 0.002993360133142995, 0.0018421805818369912, 0.0018003853116627664, 0.0017421430759043974, 0.0016295658162301671, 0.0015273250297536114, 0.0014809302001226865])
    * Topic:  6 ([21, 46, 71, 8, 26, 37, 56, 163, 146, 148], [0.02597066585114597, 0.013811418631441053, 0.013456482614800251, 0.011037439035795432, 0.010280339894431084, 0.009370728976065399, 0.008079585699476411, 0.005812665179090059, 0.005679856071011338, 0.005630616072982106])
    * Topic:  7 ([4653, 10046, 11013, 8816, 9572, 9806, 9700, 10483, 10687, 10261], [0.0013711843759944425, 0.001143158237496648, 0.0009403656297908877, 0.0009107915421267248, 0.0009029772094022746, 0.0008225093021728819, 0.0008079042902016751, 0.000683154555997964, 0.0006762717058768903, 0.0006744124221683671])
    * Topic:  8 ([79, 330, 5919, 708, 6641, 9254, 5861, 7084, 3010, 6698], [0.012258900946569532, 0.0023787483367008607, 0.0017551955336359726, 0.001706326006927757, 0.0015561123855455667, 0.0014851971460044223, 0.001470126392161258, 0.0014277937844426809, 0.0014198216015018931, 0.001417586298597661])
    * Topic:  9 ([0, 1, 2, 3, 4, 5, 6, 7, 9, 10], [0.034512767199491395, 0.011095942219650518, 0.010935419904465442, 0.00952093978841103, 0.00720447336226664, 0.0066993281268450606, 0.005440434264296813, 0.005144055585522955, 0.0045551242826680745, 0.004520414833731249])
    Topic  0
    homosexuality boswell cholesterol alzheimer cancer paul disease sin arsenokoitai hicnet 
    -----
    Topic  1
    wri markp elvis princeton lsd mantis pundurs objective cisco rusnews 
    -----
    Topic  2
    kilometers vinge balloon defamation vega stephens asa lander venera soviets 
    -----
    Topic  3
    r 0d _ p _o g 145 q k u 
    -----
    Topic  4
    wiring culture soc gfci istanbul wire neutral outlets outlet prong 
    -----
    Topic  5
    45 sleeve picture magi sweden switzerland russia promo bongo czech 
    -----
    Topic  6
    x windows dos 1 2 3 0 data image graphics 
    -----
    Topic  7
    hal9k toelle evansville uncc rti mmc den clk wawers skybridge 
    -----
    Topic  8
    00 01 aug launch meteor battan anniversary jul xxxx shower 
    -----
    Topic  9
    edu cmu com cs news srv cantaloupe net subject message 
    -----


## Recommendation


* content based filtering
* collaborative  ltering
* matrix factorization
* ALS
ALS works by iteratively solving a series of least squares regression problems.
    * rank - the number of factors in our ALS model
    * iterations - around 10 is often a good default
    * lambda - The higher the value of lambda,
the more is the regularization applied

## 문제 S-5: Spark MLib movie recommendation

* ALS로 추천


* [spark recommendation](https://www.codementor.io/spark/tutorial/building-a-recommender-with-apache-spark-python-example-app-part1)

* [spark flask](https://www.codementor.io/spark/tutorial/building-a-web-service-with-apache-spark-flask-example-app-part2)

* 주 13 - spark 추천 영화 음악? 
    * amazon similarity lookup http://blogs.gartner.com/martin-kihn/how-to-build-a-recommender-system-in-python/
    * https://www.codementor.io/spark/tutorial/building-a-web-service-with-apache-spark-flask-example-app-part2
    * https://github.com/grahamjenson/list_of_recommender_systems
        * content-based filtering
        * collaborative filtering

### S-13.1 데이터 수집

?https://inclass.kaggle.com/c/movie

* MovieLens는 University of Minnesota에서 제공하는 프로젝트, [grouplens](https://grouplens.org/)
* 영화평가 파일은 zip
* gz는 바로 읽을 수 있으나


```python
import os
import urllib

ml_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
ml_small_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

ml_fname=os.path.join(os.getcwd(),'data','ml-latest.zip')
if(not os.path.exists(ml_fname)):
    print "%s data does not exist! retrieving.." % ml_fname
    ml_f=urllib.urlretrieve(ml_url,ml_fname)

ml_small_fname=os.path.join(os.getcwd(),'data','ml-latest-small.zip')
if(not os.path.exists(ml_small_fname)):
    print "%s data does not exist! retrieving.." % ml_small_fname
    ml_small_f=urllib.urlretrieve(ml_small_url,ml_small_fname)
```

* unzip


```python
import zipfile

zipfiles=[ml_fname,ml_small_fname]
for f in zipfiles:
    zip = zipfile.ZipFile(f)
    zip.extractall('data')

```

* 압축한 파일 살펴보기


```python
!ls data/ml-latest-small/
```

    links.csv  movies.csv  ratings.csv  README.txt	tags.csv



```python
!head data/ml-latest-small/links.csv
```

    
    
    
    
    
    
    
    
    
    



```python
!head data/ml-latest-small/movies.csv
```

    
    
    
    
    
    
    
    
    
    



```python
!head data/ml-latest-small/ratings.csv
```

    
    
    
    
    
    
    
    
    
    



```python
!head data/ml-latest-small/tags.csv
```

    
    
    
    
    
    
    
    
    
    


### S-13.2 ETL

* RDD
    * 앞서 정의한 sc를 사용


```python
small_ratings = os.path.join('data', 'ml-latest-small', 'ratings.csv')

small_ratings_rdd = spark.sparkContext.textFile(small_ratings)
small_ratings_rdd_header = small_ratings_rdd.take(1)[0]
print small_ratings_rdd_header
```

    userId,movieId,rating,timestamp



```python
small_ratings_data = small_ratings_rdd\
    .filter(lambda line: line!=small_ratings_rdd_header)\
    .map(lambda line: line.split(","))\
    .map(lambda tokens: (tokens[0],tokens[1],tokens[2]))\
    .cache()
small_ratings_data.take(3)
```




    [(u'1', u'16', u'4.0'), (u'1', u'24', u'1.5'), (u'1', u'32', u'4.0')]




```python
def csvRdd(csvpath):
    _rdd = spark.sparkContext.textFile(csvpath)
    _rdd_header = _rdd.take(1)[0]
    print "header: %s" % _rdd_header
    rdd = _rdd.\
        filter(lambda line: line!=_rdd_header) \
        .map(lambda line: line.split(",")) \
        .map(lambda tokens: (tokens[0],tokens[1],tokens[2])) \
        .cache()
    return rdd
```


```python
#ratingspath = os.path.join(datapath, 'ml-latest-small', 'ratings.csv')
#ratings=csvRdd(ratingspath)
ratings=csvRdd(small_ratings)
ratings.take(3)
```

    header: userId,movieId,rating,timestamp





    [(u'1', u'16', u'4.0'), (u'1', u'24', u'1.5'), (u'1', u'32', u'4.0')]




```python
moviespath = os.path.join('data', 'ml-latest-small', 'movies.csv')
movies=csvRdd(moviespath)
movies.take(3)
```

    header: movieId,title,genres





    [(u'1', u'Toy Story (1995)', u'Adventure|Animation|Children|Comedy|Fantasy'),
     (u'2', u'Jumanji (1995)', u'Adventure|Children|Fantasy'),
     (u'3', u'Grumpier Old Men (1995)', u'Comedy|Romance')]



### S-13.3 모델링

구분 | 비율
-----|-----
train | 6
validation | 2
test | 2


```python
* MSE Mean Squared Error

* Root Mean Squared Error (RMSE)
```


```python
from pyspark.mllib.recommendation import ALS
import math

_train, _validation, _test=ratings.randomSplit([6, 2, 2], seed=0L)
_validation_01 = _validation.map(lambda x: (x[0], x[1]))
_test_01 = _test.map(lambda x: (x[0], x[1]))

seed = 5L
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(_train,rank,seed=seed,iterations=iterations,\
                      lambda_=regularization_parameter)
    predictions = model.predictAll(_validation_01)\
        .map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = _validation\
        .map(lambda r: ((int(r[0]), int(r[1])), float(r[2])))\
        .join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0]-r[1][1])**2)\
                      .mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank

```

    For rank 4 the RMSE is 0.922345252375
    For rank 8 the RMSE is 0.930922837081
    For rank 12 the RMSE is 0.925555162553
    The best model was trained with rank 4



```python
model.userFeatures
```




    <bound method MatrixFactorizationModel.userFeatures of <pyspark.mllib.recommendation.MatrixFactorizationModel object at 0x7f95e811b610>>




```python
predictions.take(3)
```




    [((384, 1084), 3.6775826431871153),
     ((668, 1084), 3.204225935944695),
     ((220, 1084), 3.8450699802260537)]



* userId, moveId를 넣으면, ratings
    * 1번 사용자, 24번 영화에 대한 rating은 약 2.04


```python
model.predict(1,24)
```




    2.0438126867172275



* 1번 사용자, 상위 10개 상품 추천


```python
model.recommendProducts(1,10)
```




    [Rating(user=1, product=2920, rating=4.922689167223983),
     Rating(user=1, product=61240, rating=4.90865915402299),
     Rating(user=1, product=86377, rating=4.905143046374245),
     Rating(user=1, product=104374, rating=4.889979338117981),
     Rating(user=1, product=79274, rating=4.860825150137431),
     Rating(user=1, product=7210, rating=4.799745725789635),
     Rating(user=1, product=3272, rating=4.758622747837389),
     Rating(user=1, product=66785, rating=4.75634224179369),
     Rating(user=1, product=3083, rating=4.749451686424952),
     Rating(user=1, product=1217, rating=4.731662442870702)]




```python

model = ALS.train(_train, best_rank, seed=seed, iterations=iterations,lambda_=regularization_parameter)
predictions = model.predictAll(_test_01).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = _test\
    .map(lambda r: ((int(r[0]), int(r[1])), float(r[2])))\
    .join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print 'For testing data the RMSE is %s' % (error)

```

    For testing data the RMSE is 0.920783817111


###



```python
Visualising London Bike Hire Journey Lengths with Python and OSRM
http://sensitivecities.com/bikeshare.html#.WUTJphPyiL4
```

### 문제 M-6: Ethereum

* kaggle


```python
base_url = Template('http://www.gutenberg.org/files/$book_id/$book_id.txt')
r = requests.get('http://www.gutenberg.org/browse/scores/top')
```
