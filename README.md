# Machine-Learning-Final-2020
 机器学习基础 大作业

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
	
- 数据说明
  **l  训练数据（weibo\_train\_data(new)）2015-02-01至2015-07-31  
  博文的全部信息都映射为一行数据。其中对用户做了一定抽样，获取了抽样用户半年的原创博文，对用户标记和博文标记做了加密 发博时间精确到天级别。

|     |     |     |
| --- | --- | --- |
| 字段  | 字段说明 | 提取说明 |
| uid | 用户标记 | 抽样&字段加密 |
| mid | 博文标记 | 抽样&字段加密 |
| time | 发博时间 | 精确到天 |
| forward_count | 博文发表一周后的转发数 |     |
| comment_count | 博文发表一周后的评论数 |     |
| like_count | 博文发表一周后的赞数 |     |
| content | 博文内容 |     |


l  预测数据（weibo\_predict\_data(new)）2015-08-01至2015-08-31

|     |     |     |
| --- | --- | --- |
| 字段  | 字段说明 | 提取说明 |
| uid | 用户标记 | 抽样&字段加密 |
| mid | 博文标记 | 抽样&字段加密 |
| time | 发博时间 | 精确到天 |
| content | 博文内容 |     |


l  选手需要提交的数据（weibo\_result\_data），选手对预测数据（weibo\_predict\_data）中每条博文一周后的转、评、赞值进行预测

|     |     |     |
| --- | --- | --- |
| 字段  | 字段说明 | 提取说明 |
| uid | 用户标记 | 抽样&字段加密 |
| mid | 博文标记 | 抽样&字段加密 |
| forward_count | 博文发表一周后的转发数 |     |
| comment_count | 博文发表一周后的评论数 |     |
| like_count | 博文发表一周后的赞数 |     |

选手提交结果文件的转、评、赞值必须为整数不接受浮点数！注意：提交格式(.txt)：uid、mid、forward\_count字段以tab键分隔，forward\_count、comment\_count、like\_count字段间以逗号分隔
