# Project Report

Dataset

Rows: 200

Target: churned（二分类，基线流失率约 27%）

Schema: 7 列（id, age, income, city, signup_date, churned, age_bin）

Cleaning

标准化文本：city 去首尾空格并统一大小写（Title Case）。

解析日期：包含 “date” 的列转为 datetime（signup_date）。

数值缺失填补：按 中位数 填补。

缺失检查：清洗后无缺失值（0%）✅

派生特征：age_bin = cut(age, [0,25,40,60,120])。

EDA Highlights

单变量分布

age：中位数 43.5，四分位数 [31, 56]（范围 18–69）。

income：中位数 ≈ 59,798，均值 ≈ 60,093，轻度右偏（最大值 ≈ 107,806）。

city：共 4 类；样本最多 Nyc（65/200）。

目标概览

全局 churned 均值 0.27（基线流失率 27%）。

age_bin 已就绪，可用于分组对比流失率。

时间维度

signup_date 为 2023-01-01 起的 200 天连续日期，可做周度/月至少两种粒度的趋势图。

数据质量

缺失率 0%，无明显结构性问题；后续重点检查异常值与高杠杆点（income 右尾）。

（可插图：eda_age_hist.png, eda_income_hist.png, eda_city_counts.png, eda_weekly_signups.png, eda_weekly_churn.png）

Model

Pipeline： 数值标准化（StandardScaler）+ 类别 One-Hot（OneHotEncoder）→ LogisticRegression

Split： 80/20，random_state=42

产物： models/pred.csv（预测）、data/clean.csv（清洗后数据）

Results

Accuracy： （在你本地运行结果填这里，例如 ACC=0.xxx）

Key charts： （将上面的 EDA 图片或 ROC/PR 曲线以文件名列出/贴图）

Takeaways

清洗后数据一致、无缺失，可直接进入建模与特征工程阶段。

income 分布右尾明显，建议采用稳健度量（如中位数/分箱）或对数变换后再与目标做关系分析。

age_bin 和 city 提供了自然的分组维度，适合做分层评估与相对风险对比。