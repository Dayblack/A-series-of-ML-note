#!/usr/bin/env python
# coding: utf-8

# # 数据预处理步骤
# 1.预处理需要导入两个库：Numpy与Pandas
# 
# 2.导入数据集：数据集一般是csv格式，每一行是一条数据记录。使用pandas.read_csv()来读取本地csv文件作为一个数据帧
# 
# 3.处理丢失数据：一般常用整列的平均值对NAN进行替换。使用sklearn.preprocessing库中的Imputer类来实现
# 
# 4.对分类数据进行解析：对于含有标签而非数值的变量，取值范围固定，类似“Yes”、“No”，我们需要将其解析为数字。使用sklearn.preprocessing库的LabelEncoder类
# 
# 5.拆分数据集为训练集合与测试集合：一般拆分比例为80：20。导入sklearn.crossvalidation库中的train_test_split()方法
# 
# 6.特征缩放：在大部分模型算法中，经常使用两点之间的欧氏距离，因而特征之间的幅度、范围、单位问题对结果影响很大/在距离计算中，会偏向于具有高幅度特征的权重。使用sklearn.preprocessing的StandardScalar类

# ## 第一步：导入库

# In[ ]:


import numpy as np
import pandas as pd


# ## 第二步：导入数据集

# In[ ]:


dataset = pd.read_csv(r'Data.csv')#读取csv文件
X = dataset.iloc[:,:-1].values#.iloc[行，列]
Y = dataset.iloc[:,3].values#第三列


# ## 第三步：处理丢失数据
# 需要注意的是，对于sklearn.preprocessing的Imputer类，调用方法为**sklearn.preprocessing.Imputer(missing_values="NaN",strategy="mean",axis=0,verbose=0,copy=True)**
# 
# 主要参数说明：
# 
# missing_values:缺失值，可以为整数或者NaN,使用"NaN"即numpy.nan
# stratrgy：替换策略，字符串，默认为"mean"
# 
# *"mean":用特征列的均值替换*
# 
# *"median":用特征列的中位数替换* 
# 
# *"most_frequence":用特征列中出现频次最高的数替换*
# 
# axis:指定轴数，axis=0代表列，axis=1代表行
# 
# ##### 注意：
# 在最新版的sklearn中，原preprocessing.Imputer被移除，此时需要调用  from sklearn.impute import SimpleImputer
# 
# ### 关于fit、fit_transfrom、transfrom个人理解：
# 1.fit和transform没有任何关系，仅仅是数据处理的两个不同环节，之所以出来fit_transform这个函数名，仅仅是为了写代码方便，会高效一点
# 
# 2.sklearn里的封装好的各种算法使用前都要fit，fit相对于整个代码而言，为后续API服务。fit之后，然后调用各种API方法，transform只是其中一个API方法，所以当你调用transform之外的方法，也必须要先fit
# 
# 3.fit原义指的是安装、使适合的意思，其实有点train的含义，但是和train不同的是，它并不是一个训练的过程，而是一个适配的过程，过程都是确定的，最后得到一个可用于转换的有价值的信息
# ##### 注意：
# 
# 1.必须先用fit_transform(trainData)，之后再transform(testData)
# 
# 2.如果直接transform(testData)，程序会报错
# 
# 3.如果fit_transfrom(trainData)后，使用fit_transform(testData)而不transform(testData)，虽然也能归一化，但是两个结果不是在同一个“标准”下的，具有明显差异。(一定要避免这种情况)
# 
# 具体细节详见：https://blog.csdn.net/weixin_38278334/article/details/82971752

# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN",strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transfrom(X[:,1:3])


# ## 第四步：解析分类数据
# **LabelEncoder**：是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码。其中包含以下常用方法：
# 
# **fit(y)** ：fit可看做一本空字典，y可看作要塞到字典中的词
# 
# **fit_transform(y)**：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值
# 
# **inverse_transform(y)**：根据索引值y获得原始数据
# 
# **transform(y)** ：将y转变成索引值，该索引值为数值，其与原数值一一对应，看以看作为字典
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencode_X = Labelencode()
X[:,0] = Labelencoder_X.fit_transform(X[:,0])


# ### 创建虚拟变量

# In[ ]:


onehotencoder =OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transfrom(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# ### 理解：preprocessing中LabelEncoder与OneHoEncoder的使用
# **1.LabelEncoder**
# 
#    用来对分类型特征值进行编码，包括文本与数值类型，其主要包括以下方法：
#    
#    fit(y1):类似字典，y1可看作要塞到字典中的词，LabelEncoder对y1中每一个不同的元素进行数值标记
#    
#    transform(y2):根据fit(y1)得到的类似的字典，对y2中每一个元素进行数值替换
#    
#    fit_transform(Y):即先fit(Y)再transform(Y),得到索引值
#    
#    inverse_transform(y)：根据索引值y获得原始数据
#    
# **2.OneHotEncoder**
# 
#    简介：有一些特征并不是以连续值的形式给出。例如：人的性别 [“male”, “female”]，来自的国家 [“from Europe”, “from US”, “from Asia”]，使  用的浏览器[“uses Firefox”, “uses Chrome”, “uses Safari”, “uses Internet Explorer”]。这种特征可以采用整数的形式进行编码，如： [“male”, “from US”, “uses Internet Explorer”] 可表示成 [0, 1, 3] ，[“female”, “from Asia”, “uses Chrome”] 可表示成[1, 2, 1]。 但是，这些整数形式的表示不能直接作为某些机器学习算法输入，因为有些机器学习算法是需要连续型的输入数据，同一列数据之间数值的大小可代表差异程度。如： [0, 1, 3]与[0,1,0]的特征差异比[0, 1, 3]与[0,1,2]之间的差异要大，但事实上它们的差异是一样的，都是浏览器使用不一样
#    
#    一个解决办法就是采用OneHotEncoder，这种表示方式将每一个分类特征变量的m个可能的取值转变成m个二值特征，对于每一条数据这m个值中仅有一个特征值为1，其他的都为0
#    
#    *下面为LabelEncoder与OneHotEncoder混合使用例子*

# In[ ]:


tmp = le.fit_transform(["n","y","o"])
print(type(tmp))
print(tmp.reshape(-1,1))
enc.fit(tmp.reshape(-1,1))
print(le.transform(["n","o"]).reshape(-1,1))
enc.transform(le.transform(["n","o"]).reshape(-1,1)).toarray()
#--------------------------------------------------------------
a=pd.DataFrame([["A",20,1.3,"Bob"],["A",21,1.4,"Alearn"],["B",22,1.4,"Clone"]])
enc.fit(le.fit_transform(a.iloc[:,0].values).reshape(-1,1)).transform(le.transform(np.array(["A","B","A"])).reshape(-1,1)).toarray()


# ## 第五步：拆分数据集为训练集合与测试集合
# 
# **train_test_split(*array,test_size=0.25,train_size=None,random_state=None,shuffle=True,stratify=None)**
# 
# *参数详解*
# 
#   **array**：切分数据源（list/np.array/pd.DataFrame/scipy_sparse matrices）
#       
#   **test_size**和**train_size**是互补和为1的一对值，为None时默认为0.25
#       
#   **shuffle**：对数据切分前是否洗牌,False不打乱样本顺序，True打乱样本顺序
#       
#   **stratify**：是否分层抽样切分数据（ If shuffle=False then stratify must be None.）stratify不为None，则保证train集合与test集合分布比例一致，可以按照比例进行分层划分，一般用在样本集中类别分布不平衡的情况下，避免因为随机采样导致某类别集中出现在某一集合而另一集合几乎没有的情况，提高后续模型在test集合上的测试效果。
#       
#   将stratify=X就是按照X中的比例分配 
# 
#   将stratify=y就是按照y中的比例分配 
#       
#   **random_set**:随机数种子，若想保持每一次实验随机取样，必须每次实验设置不同的种子，random_set=int(time.time())

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,_Y_test = train_test_split(X,Y,test_size=0.2,random_set=0)


# ## 扩展1：关于sklearn中的交叉验证
# *该部分后续会详细介绍，本部分由于涉及到train与test集合的划分，做简要介绍*
# 
# 交叉验证的目的是为了防止过拟合，同时找到最合适模型与超参数，提升模型在真实集上的稳定性
# 
# **引入**
# 
# 

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)


# 上述代码，划分了60%的数据作为训练集，剩下40%作为测试集，但有的时候为了调整超参数（比如这里SVM的C，代表SVM软间隔惩罚系数，C越大代表SVM超平面软间隔越小，对数据划分越严格），仍然有可能在测试集上过拟合。因为为了调整超参数，测试集的数据会“泄漏”给模型。为了解决这个问题，需要有一个验证集，在训练集上训练，在验证集上评估，如果表现不错的话，就可以在测试集上进行最终的评估了
# 
# 但是把数据集划分成三部分的话，训练模型的数据就大大减少了，并且结果会取决于（训练集，验证集）的随机选择
# 
# 因此就出现了交叉验证（cross-validation，简称CV）。一般是用k-fold CV，也就是k折交叉验证。训练集被划分成k个子集，每次训练的时候，用其中k-1份作为训练数据，剩下的1份作为验证，按这样重复k次。
# 
# 交叉验证计算复杂度比较高，但是充分利用了数据，对于数据集比较小的情况会有明显优势。
# 
# ### cross_val_score
# 最简单的使用交叉验证的方法是cross_val_score函数
# 
# 默认情况下，CV计算的score就是评估器的score
# 
# *scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')*评估器选用f1_macro，则cv计算评估标准也是f1_macro
# 
#    关于**f1_macro**的介绍：
# 
#        定义为f1=2(precisionrecall)/(precision+recall)
# 
#        f1越接近1越好
# 
#        定义TP(预测正确)，FP（错将其他类预测为本类），FN（本类标签预测为其他标签）
# 
#        precision=TP/(TP+FP)
# 
#        recall=TP/(TP+FN)
# 
#        在多级和多标签的情况下，是每个类别的F1分数的加权平均值。
#    
#    ***来源：sklearn.metrics.f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')***//
# 
#    *主要参数：*
# 
#    *y_ture:一维数组或标签，表示正确的标签；*
# 
#    *y_pred:分类器返回的估计标签；*
# 
#    *average可选[None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]*
#    
# 关于交叉验证所选评估器详见：https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# 
# 关于更多种类交叉验证方式详见：https://zhuanlan.zhihu.com/p/52515873
# 

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
clf = svm.SVC(kernel="linear",C=1)
scores = cross_val_score(clf,iris.data,iris.target,cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))#查看分数的均值以及分数的95%置信区间


# ## 扩展2：对数据集划分的几种不同策略
# ### KFold
# 
# K折交叉验证**sklearn.model_selection.KFold(n_splits=3, shuffle=False, random_state=None)**
# 
# *参数说明:*
# 
# n_splits：表示划分几等份
# 
# shuffle：在每次划分时，是否进行洗牌.
# 
#     若为Falses时，其效果等同于random_state等于整数，每次划分的结果相同
#     
#     若为True时，每次划分的结果都不一样，表示经过洗牌，随机取样的
#     
#         设置shuffle=False，运行两次，发现两次结果相同
# 
#         设置shuffle=True时，运行两次，发现两次运行的结果不同
# 
#         设置shuffle=True和random_state=同一个整数，发现每次运行的结果都相同
# 
# random_state：随机种子数
# 
# *方法：*
#     
# get_n_splits(X=None, y=None, groups=None)：获取参数n_splits的值
# 
# split(X, y=None, groups=None)：将数据集划分成训练集和测试集，返回**索引生成器**
# 
# 
# 
# ### StratifiedKold(常用于数据集划分时，类别不平衡的情况）
# 
# 返回分层折叠（返回每个类样本百分比实现）的KFlod变体**sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=None)**
# 
# n_splits:折叠次数，默认为3，至少为2
# 
# shuffle:是否在每次分割之前打乱顺序
# 
# random_state:随机种子，在shuffle==True时使用，默认使用np.random
# 
# *方法：*
# 
# split(X, y):
# 
# X:array-like,shape(n_sample,n_features)，训练数据集
# 
# y:array-like,shape(n_sample)，标签
# 
# **返回值：训练集数据的index与验证集数据的index**
# 
# ## ShuffleSplit
# 
# sklearn.model_selection.ShuffleSplit类用于将样本集合随机“打散”后划分为训练集、测试集(可理解为验证集，下同)，类申明如下
# 
# **sklearn.model_selection.ShuffleSplit(n_splits=10, test_size=’default’, train_size=None, random_state=None)**
# 
# n_splits:int, 划分训练集、测试集的次数，默认为10
# 
# test_size:float, int, None, default=0.1； 测试集比例或样本数量，该值为[0.0, 1.0]内的浮点数时，表示测试集占总样本的比例；该值为整型值时，表示具体的测试集样本数量；train_size不设定具体数值时，该值取默认值0.1，train_size设定具体数值时，test_size取剩余部分
# 
# random_state:int, RandomState instance or None；随机种子值，默认为None
# 
# *方法：*
# 
# split(X, y=None, groups=None)
# 
# 同上
# 
# **返回值：包含训练集、测试集索引值的迭代器**
# 
# ## GroupShuffleSplit
# 
# 作用与Shufflesplit相同，不同之处在于GroupShuffleSplit先将待分样本集分组，在划分为训练集与测试集
# **sklearn.model_selection.GroupShuffleSplit(n_splits=5, test_size=’default’, train_size=None, random_state=None)**
# 
# 参数个数及含义同ShuffleSplit，只是默认值有所不同：
# 
# n_splits:int, 划分训练集、测试集的次数，默认为5
# 
# test_size:float, int, None, default=0.1； 测试集比例或样本数量，该值为[0.0, 1.0]内的浮点数时，表示测试集占总样本的比例；该值为整型值时，表示具体的测试集样本数量；train_size不设定具体数值时，该值取默认值0.2，train_size设定具体数值时，test_size取剩余部分
# 
# random_state:int, RandomState instance or None；随机种子值，默认为None
# 
# GroupShuffleSplit类的get_n_splits、split方法与ShuffleSplit类的同名方法类似，**唯一的不同之处在于split方法的groups参数在此处生效，用于指定分组依据**
# 
# 具体划分请见：https://blog.csdn.net/hurry0808/article/details/80797969

# In[ ]:


import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# ## 第六步：特征向量化

# In[ ]:


#例1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_teat = sc_X.transform(X_test)
#例2
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
clf.score(X_test_transformed, y_test)
#可以结合管道Pipeline整合评估器
from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, iris.data, iris.target, cv=cv)

