拍拍贷比赛数据主要包括master、log_info、userupdate_info三部分数据，分别是： Master 每一行代表一个样本（一笔成功成交借款），每个样本包含200多个各类字段。 idx：每一笔贷款的unique key，可以与另外2个文件里的idx相匹配。 UserInfo_：借款人特征字段 WeblogInfo_：Info网络行为字段 Education_Info：学历学籍字段 ThirdParty_Info_PeriodN_：第三方数据时间段N字段 SocialNetwork_*：社交网络字段 LinstingInfo：借款成交时间 Target：违约标签（1 = 贷款违约，0 = 正常还款）。测试集里不包含target字段。

Log_info 借款人的登陆信息。 ListingInfo：借款成交时间 LogInfo1：操作代码 LogInfo2：操作类别 LogInfo3：登陆时间 idx：每一笔贷款的unique key

Userupdate_info 借款人修改信息 ListingInfo1：借款成交时间 UserupdateInfo1：修改内容 UserupdateInfo2：修改时间 idx：每一笔贷款的unique key