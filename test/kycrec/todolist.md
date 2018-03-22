增加
用户特征：
user_views_count
趋势特征
客户浏览去重文档数/客户总浏览次数

产品特征：
转化率特征
doc_avg_views_by_distinct_users_cf
ad_views_count

事件：
event_local_hour

平均点击率：


内容的相似度：
TF-IDF



数值特征和类别特征分开训练

由于最近产品变化不大，可尝试基于id的模型



p1:最近一个月AUM>=500W
最近一年AUM>=500w
陆金所历史最高aum>=150w
旅行者财富潜力=5 可投资产>=600万
上期末 aum>=30w

P2:
陆金所历史最高AUM 70-150W
旅行者财富潜力=4 可投资产300-600万
上期末>=30w

P3:上期末>=30w且 not in p1 p2


预约行为数据的url 3.21预约了百万活期 定期 资管计划 都点击了投资按钮
