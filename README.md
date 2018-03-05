[![Gitter](https://badges.gitter.im/smu405/s.svg)](https://gitter.im/smu405/s?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# 빅데이터 프로그래밍

* 최종수정일 20180307화

## 교과목 개요

* 본 교과목에서는 대량의 데이터를 분석하는 각 단계를 프로그래밍으로 하는 방법을 배운다.
이 과정은 우선 데이터의 수집, 데이터의 정제, 저장, 분석 및 시각화로 구분할 수
있다.
* 데이터를 수집하기 위해서는 웹사이트를 크롤링하거나, OAuth인증과 REST API를 사용할 수 있다.
수집된 데이터는 파일 또는 NoSQL을 사용하여 저장할 수 있다.
정부에서 제공하는 오픈데이터를 사용하는 방법도 실습한다.
가져온 데이터를 정제하고, 머신러닝으로 분석하고 시각화하기 위해 파이썬과 스파크를 배우게 된다.
된다. 매 강의 초반부는 주제에 필요한 개념을 설명하고, 이를 적용해 프로그래밍 과제를 풀어나가는 방식으로 진행한다.
    * 모듈1: 데이터수집 (1) 웹, (2) api
    * 모듈2: 데이터저장 file (csv, json), nosql
    * 모듈3: 분석 추론, 머신러닝
    * 모듈4: 실시간 분석 spark
    * 모듈5: 시각화

This class is to teach programming required over the steps to analyze large-scale datasets.
The students will learn to collect the data by crawling web sites and use OAuth protocols and REST API.
The data will be saved in files or NoSQL.
We will also learn to program to get open data provided by the government.
We choose to use Python and Spark for programming over the stages of data clean-up, machine learning and visualization.
Each lecture will begin with background concepts to cover the day's topics.
Then the rest of the class will be spent on lab sessions for the students to apply their learning and to complete programming exercises.

## 주별 강의 (--는 범위에서 제외한다는 뜻)

주 | 일자 | 내용
-----|-----|-----
주 1 |  3.06화 | intro 환경 Python 소개 IPython, --(Numpy, Scipy, Matplotlib, Pandas, Scikit-learn)
주 2 |  3.13화 | web crawl html dom
주 3 |  3.20화 | web crawl --regex, beautifulSoup
주 4 |  3.27화 | web crawl --xpath, css
주 5 |  4.03화 | web crawl: 테이블, wiki 
주 6 |  4.10화 | web crawl: 노래제목, 날씨, 댓글
주 7 |  4.17화 | Selenium (프로젝트 과제 1차)
주 8 |  4.24화 | 중간 시험 midterm 
주 9 |  4.01화 | 저장 file, json, --xml, csv, MongoDB
주 10 |  5.08화 | 저장 file, json, --xml, csv, MongoDB
주 11 |  5.15화 | 열린데이터 api
주 12 |  5.22화 | 석가탄신일
주 13 |  5.29화 | 열린데이터 api
주 14 |  6.05화 | spark rdd map-reduce
보강주 |  6.12화 | spark dataframe sql (12.12 프로젝트 2차 마감)
주 15 |  6.19금| 기말시험

## 과제
* 빅데이터 과제를 제안하여, 완성한다 (댓글 또는 열린데이터 사용)
* 다음 일정에 따라 ecampus에 제출한다.

주 | 기한 | 내용
-----|-----|-----
1차 | 7주 토요일 | 문제를 정하고, 어떤 데이터를 사용할 것인지. ecampus에 제출
2차 | 보강주 화요일 | 전체 제출. 문서출력 및 ecampus에 소스코드 제출. 15주차 발표.

## 참고문헌


