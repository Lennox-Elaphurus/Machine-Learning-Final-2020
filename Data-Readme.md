# Data-Readme

## 文件结构

- - 训练集：
	- 文本向量：weibo_train_data-docVector.txt
	- 标签：weibo_train_data-tag.txt

-	测试集：
	-	文本向量：weibo_predict_data-docVector.txt

## 文本向量

训练集的文本向量化，每篇博文转为一个100维32位浮点数表示的向量，没有列标签，数据按照原始数据中的顺序排列。

可直接用`dataV=np.loadtxt(fname="weibo_train_data-docVector.txt",dtype=float)`读取向量

## tag

每篇博文的转发，评论，点赞数，用一个三维int类型的向量表示，按原数据中的顺序排列属性和博文。

可直接用`tagV=np.loadtxt(fname="weibo_train_data-tag.txt",dtype=int)`读取向量