# training-log

**ver1027** **b4ns + resnext50 + b5ns + resnext50d**

| version   | private LB | public LB | model arch                                                   |
| --------- | ---------- | --------- | ------------------------------------------------------------ |
| version2  | 0.8974     | 0.9002    | final_pred2 = (final_pred1 * 0.4) + (((normalize_pred_1 * 0.45) + (normalize_pred_2 * 0.55)) * 0.6) |
| version3  | 0.8974     | 0.9005    | final_pred2 = (final_pred1 * 0.45) + (((normalize_pred_1 * 0.45) + (normalize_pred_2 * 0.55)) * 0.55) |
| version4  | 0.8975     | 0.8991    | final_pred2 = (final_pred1 * 0.3) + (((normalize_pred_1 * 0.45) + (normalize_pred_2 * 0.55)) * 0.7) |
| version5  | 0.898      | 0.9       | final_pred2 = (final_pred1 * 0.35) + (((normalize_pred_1 * 0.45) + (normalize_pred_2 * 0.55)) * 0.65) |
| version6  | 0.8984     | 0.9002    | final_pred2 = (final_pred1 * 0.35) + (((normalize_pred_1 * 0.42) + (normalize_pred_2 * 0.58)) * 0.65) |
| version7  | 0.8974     | 0.9009    | final_pred2 = (final_pred1 * 0.35) + (((normalize_pred_1 * 0.48) + (normalize_pred_2 * 0.52)) * 0.65) |
| version8  | 0.8977     | 0.9       | final_pred2 = (final_pred1 * 0.35) + (((normalize_pred_1 * 0.4) + (normalize_pred_2 * 0.6)) * 0.65) |
| version9  | 0.898      | 0.9009    | final_pred2 = (final_pred1 * 0.35) + (((normalize_pred_1 * 0.35) + (normalize_pred_2 * 0.65)) * 0.65) |
| version10 | 0.8981     | 0.8998    |                                                              |
| version11 | 0.8983     | 0.8991    |                                                              |
| version12 | 0.898      | 0.8996    |                                                              |



