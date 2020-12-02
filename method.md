机器学习基础大作业

[新浪微博互动预测-挑战Baseline](https://tianchi.aliyun.com/competition/entrance/231574/information)

- 问题：根据博主id，博文内容，发文时间预测博文的转发评论点赞数

- 方法：
	- 预处理
		- 博文内容：
       - word embedding
       - 中文分词，英文分词
		- 新增属性：
			- 	博文长度 
			- 	#标签 
			- 	超链接 
			- 	emoji
			- 博主id：字符串
		- 把三维较高的项先分离出来，记录博主id，其他的不输入
		- 博文id：没用，不输入
		- 发文时间：数值
	- 多层感知机MLP/深度神经网络DNN
