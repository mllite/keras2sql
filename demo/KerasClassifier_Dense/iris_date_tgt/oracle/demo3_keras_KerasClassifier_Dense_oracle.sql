-- This SQL code was generated by sklearn2sql (development version).
-- Copyright 2018

-- Model : KerasClassifier_Dense
-- Dataset : iris_date_tgt
-- Database : oracle


-- This SQL code can contain one or more statements, to be executed in the order they appear in this file.



-- Model deployment code

WITH keras_input AS 
(SELECT "ADS"."KEY" AS "KEY", "ADS"."Feature_0" AS "Feature_0", "ADS"."Feature_1" AS "Feature_1", "ADS"."Feature_2" AS "Feature_2", "ADS"."Feature_3" AS "Feature_3" 
FROM "IRIS_DATE_TGT" "ADS"), 
keras_input_1 AS 
(SELECT keras_input."KEY" AS "KEY", keras_input."Feature_0" AS "Feature_0", keras_input."Feature_1" AS "Feature_1", keras_input."Feature_2" AS "Feature_2", keras_input."Feature_3" AS "Feature_3" 
FROM keras_input), 
layer_dense_1 AS 
(SELECT keras_input_1."KEY" AS "KEY", 0.0 + -0.44975194334983826 * keras_input_1."Feature_0" + 0.24055200815200806 * keras_input_1."Feature_1" + 0.1637306809425354 * keras_input_1."Feature_2" + -0.474521279335022 * keras_input_1."Feature_3" AS output_1, 0.0 + -0.05272191762924194 * keras_input_1."Feature_0" + -0.41790151596069336 * keras_input_1."Feature_1" + 0.09834063053131104 * keras_input_1."Feature_2" + -0.3320915102958679 * keras_input_1."Feature_3" AS output_2, 0.0 + -0.6347334384918213 * keras_input_1."Feature_0" + -0.2240350842475891 * keras_input_1."Feature_1" + 0.11520874500274658 * keras_input_1."Feature_2" + -0.46832409501075745 * keras_input_1."Feature_3" AS output_3, 0.0 + -0.5360069274902344 * keras_input_1."Feature_0" + -0.4271448254585266 * keras_input_1."Feature_1" + 0.17860430479049683 * keras_input_1."Feature_2" + 0.06831562519073486 * keras_input_1."Feature_3" AS output_4 
FROM keras_input_1), 
activation_relu AS 
(SELECT layer_dense_1."KEY" AS "KEY", greatest(layer_dense_1.output_1, 0) AS output_1, greatest(layer_dense_1.output_2, 0) AS output_2, greatest(layer_dense_1.output_3, 0) AS output_3, greatest(layer_dense_1.output_4, 0) AS output_4 
FROM layer_dense_1), 
activation_relu_1 AS 
(SELECT activation_relu."KEY" AS "KEY", activation_relu.output_1 AS output_1, activation_relu.output_2 AS output_2, activation_relu.output_3 AS output_3, activation_relu.output_4 AS output_4 
FROM activation_relu), 
activation_relu_1_1 AS 
(SELECT activation_relu_1."KEY" AS "KEY", activation_relu_1.output_1 AS output_1, activation_relu_1.output_2 AS output_2, activation_relu_1.output_3 AS output_3, activation_relu_1.output_4 AS output_4 
FROM activation_relu_1), 
layer_dense_2 AS 
(SELECT activation_relu_1_1."KEY" AS "KEY", -0.055047694593667984 + -0.9164537787437439 * activation_relu_1_1.output_1 + -0.8462644815444946 * activation_relu_1_1.output_2 + -0.4455767869949341 * activation_relu_1_1.output_3 + 0.17921042442321777 * activation_relu_1_1.output_4 AS output_1, -0.0025790755171328783 + 0.21585404872894287 * activation_relu_1_1.output_1 + -0.3562372326850891 * activation_relu_1_1.output_2 + -0.5576794147491455 * activation_relu_1_1.output_3 + -0.6792638897895813 * activation_relu_1_1.output_4 AS output_2, 0.04657745733857155 + -0.672849178314209 * activation_relu_1_1.output_1 + -0.42303466796875 * activation_relu_1_1.output_2 + 0.09814000129699707 * activation_relu_1_1.output_3 + 0.29278361797332764 * activation_relu_1_1.output_4 AS output_3 
FROM activation_relu_1_1), 
layer_dense_2_1 AS 
(SELECT layer_dense_2."KEY" AS "KEY", layer_dense_2.output_1 AS output_1, layer_dense_2.output_2 AS output_2, layer_dense_2.output_3 AS output_3 
FROM layer_dense_2), 
score_soft_max_step1 AS 
(SELECT layer_dense_2_1."KEY" AS "KEY", layer_dense_2_1.output_1 AS "Score_0", exp(least(100.0, greatest(-100.0, layer_dense_2_1.output_1))) AS "exp_Score_0", layer_dense_2_1.output_2 AS "Score_1", exp(least(100.0, greatest(-100.0, layer_dense_2_1.output_2))) AS "exp_Score_1", layer_dense_2_1.output_3 AS "Score_2", exp(least(100.0, greatest(-100.0, layer_dense_2_1.output_3))) AS "exp_Score_2" 
FROM layer_dense_2_1), 
score_class_union_soft AS 
(SELECT soft_scu."KEY" AS "KEY", soft_scu.class AS class, soft_scu."exp_Score" AS "exp_Score" 
FROM (SELECT score_soft_max_step1."KEY" AS "KEY", 0 AS class, score_soft_max_step1."exp_Score_0" AS "exp_Score" 
FROM score_soft_max_step1 UNION ALL SELECT score_soft_max_step1."KEY" AS "KEY", 1 AS class, score_soft_max_step1."exp_Score_1" AS "exp_Score" 
FROM score_soft_max_step1 UNION ALL SELECT score_soft_max_step1."KEY" AS "KEY", 2 AS class, score_soft_max_step1."exp_Score_2" AS "exp_Score" 
FROM score_soft_max_step1) soft_scu), 
score_soft_max AS 
(SELECT score_soft_max_step1."KEY" AS "KEY", score_soft_max_step1."Score_0" AS "Score_0", score_soft_max_step1."exp_Score_0" AS "exp_Score_0", score_soft_max_step1."Score_1" AS "Score_1", score_soft_max_step1."exp_Score_1" AS "exp_Score_1", score_soft_max_step1."Score_2" AS "Score_2", score_soft_max_step1."exp_Score_2" AS "exp_Score_2", sum_exp_t."KEY_sum" AS "KEY_sum", sum_exp_t."sum_ExpScore" AS "sum_ExpScore" 
FROM score_soft_max_step1 LEFT OUTER JOIN (SELECT score_class_union_soft."KEY" AS "KEY_sum", sum(score_class_union_soft."exp_Score") AS "sum_ExpScore" 
FROM score_class_union_soft GROUP BY score_class_union_soft."KEY") sum_exp_t ON score_soft_max_step1."KEY" = sum_exp_t."KEY_sum"), 
layer_softmax AS 
(SELECT score_soft_max."KEY" AS "KEY", score_soft_max."exp_Score_0" / score_soft_max."sum_ExpScore" AS output_1, score_soft_max."exp_Score_1" / score_soft_max."sum_ExpScore" AS output_2, score_soft_max."exp_Score_2" / score_soft_max."sum_ExpScore" AS output_3 
FROM score_soft_max), 
orig_cte AS 
(SELECT layer_softmax."KEY" AS "KEY", CAST(NULL AS BINARY_DOUBLE) AS "Score_1789-07-14T00:00:00.000000000", CAST(NULL AS BINARY_DOUBLE) AS "Score_1789-08-14T00:00:00.000000000", CAST(NULL AS BINARY_DOUBLE) AS "Score_1789-09-14T00:00:00.000000000", layer_softmax.output_1 AS "Proba_1789-07-14T00:00:00.000000000", layer_softmax.output_2 AS "Proba_1789-08-14T00:00:00.000000000", layer_softmax.output_3 AS "Proba_1789-09-14T00:00:00.000000000", CAST(NULL AS BINARY_DOUBLE) AS "LogProba_1789-07-14T00:00:00.000000000", CAST(NULL AS BINARY_DOUBLE) AS "LogProba_1789-08-14T00:00:00.000000000", CAST(NULL AS BINARY_DOUBLE) AS "LogProba_1789-09-14T00:00:00.000000000", CAST(NULL AS NUMBER(19)) AS "Decision", CAST(NULL AS BINARY_DOUBLE) AS "DecisionProba" 
FROM layer_softmax), 
score_class_union AS 
(SELECT scu."KEY_u" AS "KEY_u", scu.class AS class, scu."LogProba" AS "LogProba", scu."Proba" AS "Proba", scu."Score" AS "Score" 
FROM (SELECT orig_cte."KEY" AS "KEY_u", '1789-07-14T00:00:00.000000000' AS class, orig_cte."LogProba_1789-07-14T00:00:00.000000000" AS "LogProba", orig_cte."Proba_1789-07-14T00:00:00.000000000" AS "Proba", orig_cte."Score_1789-07-14T00:00:00.000000000" AS "Score" 
FROM orig_cte UNION ALL SELECT orig_cte."KEY" AS "KEY_u", '1789-08-14T00:00:00.000000000' AS class, orig_cte."LogProba_1789-08-14T00:00:00.000000000" AS "LogProba", orig_cte."Proba_1789-08-14T00:00:00.000000000" AS "Proba", orig_cte."Score_1789-08-14T00:00:00.000000000" AS "Score" 
FROM orig_cte UNION ALL SELECT orig_cte."KEY" AS "KEY_u", '1789-09-14T00:00:00.000000000' AS class, orig_cte."LogProba_1789-09-14T00:00:00.000000000" AS "LogProba", orig_cte."Proba_1789-09-14T00:00:00.000000000" AS "Proba", orig_cte."Score_1789-09-14T00:00:00.000000000" AS "Score" 
FROM orig_cte) scu), 
score_max AS 
(SELECT orig_cte."KEY" AS "KEY", orig_cte."Score_1789-07-14T00:00:00.000000000" AS "Score_1789-07-14T00:00:0_1", orig_cte."Score_1789-08-14T00:00:00.000000000" AS "Score_1789-08-14T00:00:0_2", orig_cte."Score_1789-09-14T00:00:00.000000000" AS "Score_1789-09-14T00:00:0_3", orig_cte."Proba_1789-07-14T00:00:00.000000000" AS "Proba_1789-07-14T00:00:0_4", orig_cte."Proba_1789-08-14T00:00:00.000000000" AS "Proba_1789-08-14T00:00:0_5", orig_cte."Proba_1789-09-14T00:00:00.000000000" AS "Proba_1789-09-14T00:00:0_6", orig_cte."LogProba_1789-07-14T00:00:00.000000000" AS "LogProba_1789-07-14T00:0_7", orig_cte."LogProba_1789-08-14T00:00:00.000000000" AS "LogProba_1789-08-14T00:0_8", orig_cte."LogProba_1789-09-14T00:00:00.000000000" AS "LogProba_1789-09-14T00:0_9", orig_cte."Decision" AS "Decision", orig_cte."DecisionProba" AS "DecisionProba", max_select."KEY_m" AS "KEY_m", max_select."max_Proba" AS "max_Proba" 
FROM orig_cte LEFT OUTER JOIN (SELECT score_class_union."KEY_u" AS "KEY_m", max(score_class_union."Proba") AS "max_Proba" 
FROM score_class_union GROUP BY score_class_union."KEY_u") max_select ON orig_cte."KEY" = max_select."KEY_m"), 
union_with_max AS 
(SELECT score_class_union."KEY_u" AS "KEY_u", score_class_union.class AS class, score_class_union."LogProba" AS "LogProba", score_class_union."Proba" AS "Proba", score_class_union."Score" AS "Score", score_max."KEY" AS "KEY", score_max."Score_1789-07-14T00:00:0_1" AS "Score_1789-07-14T00:00:0_1", score_max."Score_1789-08-14T00:00:0_2" AS "Score_1789-08-14T00:00:0_2", score_max."Score_1789-09-14T00:00:0_3" AS "Score_1789-09-14T00:00:0_3", score_max."Proba_1789-07-14T00:00:0_4" AS "Proba_1789-07-14T00:00:0_4", score_max."Proba_1789-08-14T00:00:0_5" AS "Proba_1789-08-14T00:00:0_5", score_max."Proba_1789-09-14T00:00:0_6" AS "Proba_1789-09-14T00:00:0_6", score_max."LogProba_1789-07-14T00:0_7" AS "LogProba_1789-07-14T00:0_7", score_max."LogProba_1789-08-14T00:0_8" AS "LogProba_1789-08-14T00:0_8", score_max."LogProba_1789-09-14T00:0_9" AS "LogProba_1789-09-14T00:0_9", score_max."Decision" AS "Decision", score_max."DecisionProba" AS "DecisionProba", score_max."KEY_m" AS "KEY_m", score_max."max_Proba" AS "max_Proba" 
FROM score_class_union LEFT OUTER JOIN score_max ON score_class_union."KEY_u" = score_max."KEY"), 
arg_max_cte AS 
(SELECT score_max."KEY" AS "KEY", score_max."Score_1789-07-14T00:00:0_1" AS "Score_1789-07-14T00:00:0_1", score_max."Score_1789-08-14T00:00:0_2" AS "Score_1789-08-14T00:00:0_2", score_max."Score_1789-09-14T00:00:0_3" AS "Score_1789-09-14T00:00:0_3", score_max."Proba_1789-07-14T00:00:0_4" AS "Proba_1789-07-14T00:00:0_4", score_max."Proba_1789-08-14T00:00:0_5" AS "Proba_1789-08-14T00:00:0_5", score_max."Proba_1789-09-14T00:00:0_6" AS "Proba_1789-09-14T00:00:0_6", score_max."LogProba_1789-07-14T00:0_7" AS "LogProba_1789-07-14T00:0_7", score_max."LogProba_1789-08-14T00:0_8" AS "LogProba_1789-08-14T00:0_8", score_max."LogProba_1789-09-14T00:0_9" AS "LogProba_1789-09-14T00:0_9", score_max."Decision" AS "Decision", score_max."DecisionProba" AS "DecisionProba", score_max."KEY_m" AS "KEY_m", score_max."max_Proba" AS "max_Proba", "arg_max_t_Proba"."KEY_Proba" AS "KEY_Proba", "arg_max_t_Proba"."arg_max_Proba" AS "arg_max_Proba" 
FROM score_max LEFT OUTER JOIN (SELECT union_with_max."KEY" AS "KEY_Proba", max(union_with_max.class) AS "arg_max_Proba" 
FROM union_with_max 
WHERE union_with_max."max_Proba" <= union_with_max."Proba" GROUP BY union_with_max."KEY") "arg_max_t_Proba" ON score_max."KEY" = "arg_max_t_Proba"."KEY_Proba")
 SELECT arg_max_cte."KEY" AS "KEY", arg_max_cte."Score_1789-07-14T00:00:0_1" AS "Score_1789-07-14T00:00:00.000000000", arg_max_cte."Score_1789-08-14T00:00:0_2" AS "Score_1789-08-14T00:00:00.000000000", arg_max_cte."Score_1789-09-14T00:00:0_3" AS "Score_1789-09-14T00:00:00.000000000", arg_max_cte."Proba_1789-07-14T00:00:0_4" AS "Proba_1789-07-14T00:00:00.000000000", arg_max_cte."Proba_1789-08-14T00:00:0_5" AS "Proba_1789-08-14T00:00:00.000000000", arg_max_cte."Proba_1789-09-14T00:00:0_6" AS "Proba_1789-09-14T00:00:00.000000000", CASE WHEN (arg_max_cte."Proba_1789-07-14T00:00:0_4" IS NULL OR arg_max_cte."Proba_1789-07-14T00:00:0_4" > 0.0) THEN ln(arg_max_cte."Proba_1789-07-14T00:00:0_4") ELSE -BINARY_DOUBLE_INFINITY END AS "LogProba_1789-07-14T00:00:00.000000000", CASE WHEN (arg_max_cte."Proba_1789-08-14T00:00:0_5" IS NULL OR arg_max_cte."Proba_1789-08-14T00:00:0_5" > 0.0) THEN ln(arg_max_cte."Proba_1789-08-14T00:00:0_5") ELSE -BINARY_DOUBLE_INFINITY END AS "LogProba_1789-08-14T00:00:00.000000000", CASE WHEN (arg_max_cte."Proba_1789-09-14T00:00:0_6" IS NULL OR arg_max_cte."Proba_1789-09-14T00:00:0_6" > 0.0) THEN ln(arg_max_cte."Proba_1789-09-14T00:00:0_6") ELSE -BINARY_DOUBLE_INFINITY END AS "LogProba_1789-09-14T00:00:00.000000000", arg_max_cte."arg_max_Proba" AS "Decision", arg_max_cte."max_Proba" AS "DecisionProba" 
FROM arg_max_cte