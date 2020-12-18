# Data-Readme

- 训练集：
	- 数据特征：weibo_train_data-docVector.txt
	- 标签：

-	测试集：
	-	weibo_predict_data-docVector.txt


训练集的文本向量化，每篇博文转为一个100维32位浮点数表示的向量，没有列标签，数据按照原始数据中的顺序排列。

可直接用`dataV=np.loadtxt(fname="midData/weibo_train_data-docVector.txt",dtype=float)`读取向量

