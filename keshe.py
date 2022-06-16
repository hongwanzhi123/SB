# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
# 数据预处理
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df
# 读取数据
data = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
data = reduce_mem_usage(data)
# 查看数据
print(data.shape)
# 数据可视化
plt.figure(1)
for i in range(4):
  plt.plot(range(data.shape[0])[:512], data.iloc[:,i].values[:512])
  plt.show()
# 按签拆分数据
SIGNALS = []
for i in range(4):
  signal = data.iloc[:,i].values
  nan_array = np.isnan(signal)
  not_nan_array = ~nan_array
  new_signal = signal[not_nan_array]
  SIGNALS.append(new_signal)
print(SIGNALS[0].shape)
lengths = [SIGNALS[i].shape[0] for i in range(4)]
# 基于train.csv文件生成的样本
X_test = pd.read_csv('test.csv')
columns_sample = list(X_test.columns)[1:] + ['label']
def sample_generater(SIGNALS, size, columns_sample):
  data_reset = []
  for i in range(4):
    signal_i = SIGNALS[i]
    m = random.choice(range(size))
    print(m)
    indexs_i = range(m, m + size*(int(len(signal_i)/size)-2), int(size/10))  # 此处的5可以控制样本量
    for j in indexs_i:
      sample_ = list(signal_i[j:j+size]) + [i]
      data_reset.append(sample_)
  data_reset = pd.DataFrame(data_reset, columns=columns_sample)
  print(data_reset.shape)
  return data_reset

data_reset = sample_generater(SIGNALS, size=512, columns_sample=columns_sample)
from sklearn.utils import shuffle
data_reset = shuffle(data_reset)
data_reset.to_csv('train_samples.csv', index=False)
print(data_reset.head())
temp = pd.read_csv('train_samples.csv')
print(temp.isnull().any())  # 判断有没有空值
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
# 加载模型
model = lgb.Booster(model_file='lightGBM_model_0501.txt')
X_test_features = pd.read_csv('X_test_features.csv')
test_pre_lgb = model.predict(X_test_features, num_iteration=model.best_iteration)
preds = np.argmax(test_pre_lgb, axis=1)
submit = pd.read_csv('submit_sample.csv')
submit['label'] = preds
submit.to_csv('submit.csv', index=False)
# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
data_features_labels = pd.read_csv('Data_features.csv')
data_features = data_features_labels.iloc[:,:-1].values
data_labels = data_features_labels.iloc[:,-1].values
X_train, X_val, y_train, y_val = train_test_split(data_features, data_labels, random_state=501) # 0501

# 数据准备
train_matrix = lgb.Dataset(X_train, label=y_train)
valid_matrix = lgb.Dataset(X_val, label=y_val)


# 构建lightgbm
params = {
    'learning_rate': 0.1,
    'boosting': 'gbdt',
    'lambda_l2': 0.1,
    'max_depth': -1,
    'num_leaves': 512,
    'bagging_fraction': 0.8,
    'feature_fraction':0.8,
    'metric': None,
    'objective': 'multiclass',
    'num_class': 4,
    'nthread': 10,
    'verbose': -1,
}

# 使用lightgbm训练
model = lgb.train(params,
          train_set=train_matrix,
          valid_sets=valid_matrix,
          num_boost_round=2000,   # 决策树提升循环次数
          verbose_eval=50,
          early_stopping_rounds=200,
          # feval=f1_score
          )

# 对验证集进行预测
'''对验证集进行预测'''
val_pre_lgb = model.predict(X_val, num_iteration=model.best_iteration)
preds = np.argmax(val_pre_lgb, axis=1)
score = f1_score(y_true=y_val, y_pred=preds, average='macro')
print('未调参前lightgbm单模型在验证集上的f1：{}'.format(score))

# 对模型进行保存
model.save_model('lightGBM_model_0501.txt')
# -*- coding:utf-8 -*-
import tsfel
import pandas as pd

# 加载数据
path_file = 'train_samples.csv'
data_ = pd.read_csv(path_file)

# 必要的话，利用时频域分析手段构造一些特征用于分类。
data_features = pd.DataFrame()
data_labels = []
for i in range(data_.shape[0]):
  signal_i = data_.iloc[i,:-1]
  labels_i = data_.iloc[i,-1]
  cfg_file = tsfel.get_featuuuures_by_domain()
  features_i = tsfel.time_series_features_extractor(cfg_file, signal_i, fs=1, window_size=512)  #非常耗时
  data_features = pd.concat([data_features, features_i])
  data_labels.append(int(labels_i))
  # if i==10:
  #   break

data_features['label'] = data_labels
print(data_features.head(10))
data_features.to_csv('Data_features.csv')
# -*- coding:utf-8 -*-
import tsfel
import pandas as pd
# 对测试集进行预测
X_test = pd.read_csv('test.csv')
# 特征工程

X_test_features = pd.DataFrame()
for i in range(X_test.shape[0]):
  signal_i = X_test.iloc[i,1:]
  cfg_file = tsfel.get_features_by_domain()
  features_i = tsfel.time_series_features_extractor(cfg_file, signal_i, fs=1, window_size=512)  #非常耗时
  X_test_features = pd.concat([X_test_features, features_i])
  # if i==10:
  #   break

print(X_test_features)
X_test_features.to_csv('X_test_features.csv')
