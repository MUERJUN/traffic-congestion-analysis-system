# M3 验收记录

## 结论

M3三模型训练、统一评价、模型选择和基础解释已完成，可以进入Dashboard对接和系统集成阶段。

## 模型要求

- [x] Logistic Regression完成。
- [x] Random Forest完成。
- [x] XGBoost完成。
- [x] 三模型使用同一标签、特征、Validation和Test口径。
- [x] 输出Accuracy、Precision、Recall、F1-score、ROC-AUC。
- [x] Validation选择阈值，Test只做最终评价。
- [x] 按拥堵类F1、Recall、ROC-AUC选择最终模型。

## 结果要求

- [x] 输出模型比较CSV和JSON。
- [x] 输出混淆矩阵和ROC曲线。
- [x] 输出三模型全局特征重要性。
- [x] 输出最终模型单次高风险样本解释。
- [x] 记录Train抽样上限和随机种子。
- [x] 记录类别不平衡处理。

## 防泄漏与文件检查

- [x] 预处理只在Train样本拟合。
- [x] 测试集未用于调参或阈值选择。
- [x] 模型输入字段与M2字段字典一致。
- [x] 三个模型文件可加载。
- [x] 模型产物受`.gitignore`保护，未提交二进制。
- [x] 13项M2测试继续通过。

## 最终选择

XGBoost：Validation F1=0.8644，Test F1=0.8651，Test Recall=0.8754，Test ROC-AUC=0.9887。

下一阶段应由Track B读取固定版本的`model_comparison.csv`、`local_explanation.json`、最终模型和M2数据摘要，不在页面中重新训练或修改阈值。

