# subject-classifier

目的：对文本数据做学科分类

## 训练学科分类器

步骤：(1) 构造训练和测试数据集；(2) 在训练集上训练学科分类模型，在测试集上计算准确率；(3) 采用模型预测文本数据所属学科。

代码在classifier文件夹，依赖环境：

    pip install scikit-learn fasttext

### 数据集

训练数据来自[web of science](https://www.webofscience.com/wos/woscc/basic-search)提供的论文摘要。[Web of Science Core Collection](https://webofscience.help.clarivate.com/Content/wos-core-collection/wos-core-collection.htm)中的Subject Categories提供了学科类别，每篇论文已经打上了学科标签。对每个学科，按照与学科的相关性由高到低排序，下载前10万篇论文的信息，少数冷门学科总论文数不到10万篇。论文信息包含标题、摘要、关键词、作者等。最终整理出包含233个学科的json文件，在压缩文件[wos_json.rar](https://huggingface.co/datasets/lzy0928/wos_json)中，解压文件：

    unrar x wos_json.rar

进入classifier文件夹：

    cd classifier

将目标学科的论文摘要作为正样例，其它学科取前1万篇合在一起作为负样例：

    python data_train_test.py -dd '../wos_json' -od './data_train_test' -n 'Engineering,_Aerospace'

其中 -dd 是数据集路径，-od是输出的训练测试集路径，-n 是目标学科，这里以航空航天工程为例。

### 训练模型

采用[fasttext](https://github.com/facebookresearch/fastText)训练二分类器：

    python fasttext_model.py -dd './data_train_test' -md './models' -mn 'subject_aerospace.model'

其中 -dd 是训练测试集路径，-md 是模型保存路径，-mn 是模型名。

### 模型推理

导入训练好的模型，预测文本数据所属学科：

    python fasttext_predict.py

输出是fasttext_predict.py文件中，text文本所属学科和概率。