{\rtf1\ansi\ansicpg1252\cocoartf1348\cocoasubrtf170
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
import os\
import xlrd\
import urllib\
import h2o\
\
\
\
#set your path here\
os.chdir('/yourpath')\
\
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'\
filename='spamdata.data'\
urllib.urlretrieve(url, filename)\
h2o.init()           \
\
\
\
spamdata = h2o.import_file(os.path.realpath("/Users/Quandbee1/Desktop/pthw/spamdata.data"))\
spamdata['C58']=spamdata['C58'].asfactor()\
train, valid, test= spamdata.split_frame([0.6,.2], seed=1234)\
\
spam_X = spamdata.col_names[:-1]    \
spam_Y = spamdata.col_names[-1]\
\
\
from h2o.estimators.gbm import H2OGradientBoostingEstimator\
from h2o.estimators.random_forest import H2ORandomForestEstimator\
from h2o.grid.grid_search import H2OGridSearch\
\
\
hyper_parameters=\{'ntrees':[300], 'max_depth':[3,6,10,12,50],'balance_classes':['True','False'],'sample_rate':[.5,.6,.8,.9]\}\
grid_search = H2OGridSearch(H2ORandomForestEstimator, hyper_params=hyper_parameters)\
grid_search.train(x=spam_X, y=spam_Y,training_frame=train)\
print 'this is the optimum solution for hyper parameters search %s' % grid_search.show()\
\
final = H2ORandomForestEstimator(ntrees=300, max_depth=50,balance_classes=True,sample_rate=.9)\
final.train(x=spam_X, y=spam_Y,training_frame=train)\
print final.predict(test)\
\
hyper_parameters=\{'ntrees':[300],'max_depth':[12,30,50],'sample_rate':[.5,.7,1],'col_sample_rate':[.9,1],\
'learn_rate':[.01,.1,.3],\}\
grid_search = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params=hyper_parameters)\
grid_search.train(x=spam_X, y=spam_Y, training_frame=train)\
print 'this is the optimum solution for hyper parameters search %s' % grid_search.show()\
\
spam_gbm2 = H2OGradientBoostingEstimator(\
  ntrees=300,\
  learn_rate=0.3,\
  max_depth=30,\
  sample_rate=1,\
  col_sample_rate=0.9,\
  score_each_iteration=True,\
  seed=2000000\
)\
spam_gbm2.train(spam_X, spam_Y, training_frame=train, validation_frame=valid)\
\
confusion_matrix = spam_gbm2.confusion_matrix(metrics="accuracy")\
print confusion_matrix\
print spam_gbm2.score_history()\
print spam_gbm2.predict(test)}