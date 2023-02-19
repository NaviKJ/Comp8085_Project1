import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics # is used to create classification results
from six import StringIO # is used for plotting the decision tree
from IPython.display import Image # is used for plotting the decision tree
from IPython.core.display import HTML # is used for showing the confusion matrix
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from pandas import factorize
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

column_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
'trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label']
pdata = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv",header=None, names=column_names, skiprows=1, dtype='unicode')

df = pd.DataFrame(pdata, columns=column_names)

del df['ct_flw_http_mthd']
del df['is_ftp_login']
del df['ct_ftp_cmd']

df['proto']=pd.factorize(df['proto'])[0]
df['state']=pd.factorize(df['state'])[0]
df['dsport']=pd.factorize(df['dsport'])[0]
df['srcip']=pd.factorize(df['srcip'])[0]
df['sport']=pd.factorize(df['sport'])[0]
df['dstip']=pd.factorize(df['dstip'])[0]
df['dur']=pd.factorize(df['dur'])[0]
df['service']=pd.factorize(df['service'])[0]

df["service"].replace('-','None')
df["attack_cat"].fillna('None', inplace = True)

feature_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
'trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl',
'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']

def attack_class():
  X = df[feature_cols] # Features
  y = df.attack_cat # Target variable


  X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
  X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)

  X_train.apply(LabelEncoder().fit_transform)
  X_test.apply(LabelEncoder().fit_transform)
  preprocessing.normalize(X_train)

  knnClassifier=KNeighborsClassifier()
  knnClassifier.fit(X_train, y_train)

  y_pred = knnClassifier.predict(X_validation)
  confusion_matrix(y_validation, y_pred)
  accuracy_score(y_validation, y_pred)
  print("Accuracy before feature selection for target 'attack_cat': {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
  print(metrics.classification_report(y_validation, y_pred))
  micro_f1 = f1_score(y_validation, y_pred, average='micro')
  print("Micro F1 Score before feature selection for target 'attack_cat':", micro_f1) 

  KBest = SelectKBest(chi2, k=10).fit(X, y) 

  f = KBest.get_support(1) #the most important features

  X_new = X[X.columns[f]] # final features

  #Out of these 10 imporatant features, I adjusted further to train the model to get desired accuracy, hence defining top 4 feature columns as predicted by Kbest and some adjustments
  
  feature_cols2 = ['dbytes','sbytes','dsport','sttl']

  X = df[feature_cols2] # Features
  y = df.attack_cat # Target variable

  X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
  X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)

  X_train.apply(LabelEncoder().fit_transform)
  X_test.apply(LabelEncoder().fit_transform)
  preprocessing.normalize(X_train)

  knnClassifier=KNeighborsClassifier()
  knnClassifier.fit(X_train, y_train)

  y_pred = knnClassifier.predict(X_validation)
  confusion_matrix(y_validation, y_pred)
  accuracy_score(y_validation, y_pred)
  print("Accuracy after feature selection for target 'attack_cat': {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
  print(metrics.classification_report(y_validation, y_pred))
  micro_f1 = f1_score(y_validation, y_pred, average='micro')
  print("Micro F1 Score after feature selection for target 'attack_cat':", micro_f1) 

def label_class():

  X = df[feature_cols] # Features
  y = df.Label # Target variable


  X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
  X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)

  X_train.apply(LabelEncoder().fit_transform)
  X_test.apply(LabelEncoder().fit_transform)
  preprocessing.normalize(X_train)

  knnClassifier=KNeighborsClassifier()
  knnClassifier.fit(X_train, y_train)

  y_pred = knnClassifier.predict(X_validation)
  confusion_matrix(y_validation, y_pred)
  accuracy_score(y_validation, y_pred)
  print("Accuracy before feature selection for target 'Label': {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
  print(metrics.classification_report(y_validation, y_pred))
  micro_f1 = f1_score(y_validation, y_pred, average='micro')
  print("Micro F1 Score before feature selection for target 'Label':", micro_f1) 
 
  KBest = SelectKBest(chi2, k=10).fit(X, y) 

  f = KBest.get_support(1) #the most important features

  X_new = X[X.columns[f]] # final features

  X_new

  X = X_new
  y = df.Label # Target variable

  X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
  X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)

  X_train.apply(LabelEncoder().fit_transform)
  X_test.apply(LabelEncoder().fit_transform)
  preprocessing.normalize(X_train)

  knnClassifier=KNeighborsClassifier()
  knnClassifier.fit(X_train, y_train)

  y_pred = knnClassifier.predict(X_validation)
  confusion_matrix(y_validation, y_pred)
  accuracy_score(y_validation, y_pred)
  print("Accuracy after feature selection for target 'Label': {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
  print(metrics.classification_report(y_validation, y_pred))
  micro_f1 = f1_score(y_validation, y_pred, average='micro')
  print("Micro F1 Score after feature selection for target 'Label':", micro_f1) 

label_class()

attack_class()