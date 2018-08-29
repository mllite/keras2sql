-- This SQL code was generated by sklearn2sql (development version).
-- Copyright 2018

-- Model : KerasClassifier_Dense
-- Dataset : FourClass_10
-- Database : monetdb


-- This SQL code can contain one or more statements, to be executed in the order they appear in this file.



-- Model deployment code

WITH keras_input AS 
(SELECT "ADS"."KEY" AS "KEY", "ADS"."Feature_0" AS "Feature_0", "ADS"."Feature_1" AS "Feature_1", "ADS"."Feature_2" AS "Feature_2", "ADS"."Feature_3" AS "Feature_3", "ADS"."Feature_4" AS "Feature_4", "ADS"."Feature_5" AS "Feature_5", "ADS"."Feature_6" AS "Feature_6", "ADS"."Feature_7" AS "Feature_7", "ADS"."Feature_8" AS "Feature_8", "ADS"."Feature_9" AS "Feature_9" 
FROM "FourClass_10" AS "ADS"), 
keras_input_1 AS 
(SELECT keras_input."KEY" AS "KEY", keras_input."Feature_0" AS "Feature_0", keras_input."Feature_1" AS "Feature_1", keras_input."Feature_2" AS "Feature_2", keras_input."Feature_3" AS "Feature_3", keras_input."Feature_4" AS "Feature_4", keras_input."Feature_5" AS "Feature_5", keras_input."Feature_6" AS "Feature_6", keras_input."Feature_7" AS "Feature_7", keras_input."Feature_8" AS "Feature_8", keras_input."Feature_9" AS "Feature_9" 
FROM keras_input), 
layer_dense_1 AS 
(SELECT keras_input_1."KEY" AS "KEY", -0.04993262141942978 + -0.2021024227142334 * keras_input_1."Feature_0" + -0.42280685901641846 * keras_input_1."Feature_1" + -0.36015793681144714 * keras_input_1."Feature_2" + -0.041913799941539764 * keras_input_1."Feature_3" + 0.33155015110969543 * keras_input_1."Feature_4" + 0.5227947235107422 * keras_input_1."Feature_5" + -0.033317454159259796 * keras_input_1."Feature_6" + 0.5033979415893555 * keras_input_1."Feature_7" + 0.5297062993049622 * keras_input_1."Feature_8" + -0.3236433267593384 * keras_input_1."Feature_9" AS output_1, -0.13578669726848602 + 0.45822301506996155 * keras_input_1."Feature_0" + 0.017651036381721497 * keras_input_1."Feature_1" + 0.0020273148547858 * keras_input_1."Feature_2" + -0.6170789003372192 * keras_input_1."Feature_3" + -0.15927369892597198 * keras_input_1."Feature_4" + -0.20706449449062347 * keras_input_1."Feature_5" + -0.30327311158180237 * keras_input_1."Feature_6" + -0.09646669775247574 * keras_input_1."Feature_7" + 0.1871163696050644 * keras_input_1."Feature_8" + 0.05328650772571564 * keras_input_1."Feature_9" AS output_2, -0.029154956340789795 + 0.02302791178226471 * keras_input_1."Feature_0" + 0.6142361164093018 * keras_input_1."Feature_1" + 0.4750308096408844 * keras_input_1."Feature_2" + -0.4363521635532379 * keras_input_1."Feature_3" + -0.21920470893383026 * keras_input_1."Feature_4" + -0.4923766255378723 * keras_input_1."Feature_5" + 0.06177271157503128 * keras_input_1."Feature_6" + 0.44862836599349976 * keras_input_1."Feature_7" + -0.10622666031122208 * keras_input_1."Feature_8" + -0.2312604784965515 * keras_input_1."Feature_9" AS output_3, 0.04594492167234421 + -0.2709537744522095 * keras_input_1."Feature_0" + -0.39040815830230713 * keras_input_1."Feature_1" + -0.42413395643234253 * keras_input_1."Feature_2" + 0.4746885299682617 * keras_input_1."Feature_3" + 0.6436420679092407 * keras_input_1."Feature_4" + 0.1690145581960678 * keras_input_1."Feature_5" + -0.2205279916524887 * keras_input_1."Feature_6" + -0.24197648465633392 * keras_input_1."Feature_7" + -0.393185555934906 * keras_input_1."Feature_8" + 0.3349796235561371 * keras_input_1."Feature_9" AS output_4 
FROM keras_input_1), 
activation_relu AS 
(SELECT layer_dense_1."KEY" AS "KEY", CASE WHEN (layer_dense_1.output_1 <= 0) THEN 0 ELSE layer_dense_1.output_1 END AS output_1, CASE WHEN (layer_dense_1.output_2 <= 0) THEN 0 ELSE layer_dense_1.output_2 END AS output_2, CASE WHEN (layer_dense_1.output_3 <= 0) THEN 0 ELSE layer_dense_1.output_3 END AS output_3, CASE WHEN (layer_dense_1.output_4 <= 0) THEN 0 ELSE layer_dense_1.output_4 END AS output_4 
FROM layer_dense_1), 
activation_relu_1 AS 
(SELECT activation_relu."KEY" AS "KEY", activation_relu.output_1 AS output_1, activation_relu.output_2 AS output_2, activation_relu.output_3 AS output_3, activation_relu.output_4 AS output_4 
FROM activation_relu), 
activation_relu_1_1 AS 
(SELECT activation_relu_1."KEY" AS "KEY", activation_relu_1.output_1 AS output_1, activation_relu_1.output_2 AS output_2, activation_relu_1.output_3 AS output_3, activation_relu_1.output_4 AS output_4 
FROM activation_relu_1), 
layer_dense_2 AS 
(SELECT activation_relu_1_1."KEY" AS "KEY", -0.01243742648512125 + 0.5128546357154846 * activation_relu_1_1.output_1 + -0.5058655738830566 * activation_relu_1_1.output_2 + -0.3671221137046814 * activation_relu_1_1.output_3 + 0.04668010026216507 * activation_relu_1_1.output_4 AS output_1, -0.04768518730998039 + -0.6234007477760315 * activation_relu_1_1.output_1 + 0.045032255351543427 * activation_relu_1_1.output_2 + 0.4655691981315613 * activation_relu_1_1.output_3 + 0.3210568428039551 * activation_relu_1_1.output_4 AS output_2, -0.11574915796518326 + 0.6564697027206421 * activation_relu_1_1.output_1 + 0.7741883397102356 * activation_relu_1_1.output_2 + -0.4866999089717865 * activation_relu_1_1.output_3 + -0.6872887015342712 * activation_relu_1_1.output_4 AS output_3, 0.17133234441280365 + -0.3738657534122467 * activation_relu_1_1.output_1 + -0.24590381979942322 * activation_relu_1_1.output_2 + -0.06052662059664726 * activation_relu_1_1.output_3 + -0.24239271879196167 * activation_relu_1_1.output_4 AS output_4 
FROM activation_relu_1_1), 
layer_dense_2_1 AS 
(SELECT layer_dense_2."KEY" AS "KEY", layer_dense_2.output_1 AS output_1, layer_dense_2.output_2 AS output_2, layer_dense_2.output_3 AS output_3, layer_dense_2.output_4 AS output_4 
FROM layer_dense_2), 
score_soft_max_step1 AS 
(SELECT layer_dense_2_1."KEY" AS "KEY", layer_dense_2_1.output_1 AS "Score_0", exp(CASE WHEN (100.0 <= CASE WHEN (-100.0 <= layer_dense_2_1.output_1) THEN layer_dense_2_1.output_1 ELSE -100.0 END) THEN 100.0 ELSE CASE WHEN (-100.0 <= layer_dense_2_1.output_1) THEN layer_dense_2_1.output_1 ELSE -100.0 END END) AS "exp_Score_0", layer_dense_2_1.output_2 AS "Score_1", exp(CASE WHEN (100.0 <= CASE WHEN (-100.0 <= layer_dense_2_1.output_2) THEN layer_dense_2_1.output_2 ELSE -100.0 END) THEN 100.0 ELSE CASE WHEN (-100.0 <= layer_dense_2_1.output_2) THEN layer_dense_2_1.output_2 ELSE -100.0 END END) AS "exp_Score_1", layer_dense_2_1.output_3 AS "Score_2", exp(CASE WHEN (100.0 <= CASE WHEN (-100.0 <= layer_dense_2_1.output_3) THEN layer_dense_2_1.output_3 ELSE -100.0 END) THEN 100.0 ELSE CASE WHEN (-100.0 <= layer_dense_2_1.output_3) THEN layer_dense_2_1.output_3 ELSE -100.0 END END) AS "exp_Score_2", layer_dense_2_1.output_4 AS "Score_3", exp(CASE WHEN (100.0 <= CASE WHEN (-100.0 <= layer_dense_2_1.output_4) THEN layer_dense_2_1.output_4 ELSE -100.0 END) THEN 100.0 ELSE CASE WHEN (-100.0 <= layer_dense_2_1.output_4) THEN layer_dense_2_1.output_4 ELSE -100.0 END END) AS "exp_Score_3" 
FROM layer_dense_2_1), 
score_class_union_soft AS 
(SELECT soft_scu."KEY" AS "KEY", soft_scu.class AS class, soft_scu."exp_Score" AS "exp_Score" 
FROM (SELECT score_soft_max_step1."KEY" AS "KEY", 0 AS class, score_soft_max_step1."exp_Score_0" AS "exp_Score" 
FROM score_soft_max_step1 UNION ALL SELECT score_soft_max_step1."KEY" AS "KEY", 1 AS class, score_soft_max_step1."exp_Score_1" AS "exp_Score" 
FROM score_soft_max_step1 UNION ALL SELECT score_soft_max_step1."KEY" AS "KEY", 2 AS class, score_soft_max_step1."exp_Score_2" AS "exp_Score" 
FROM score_soft_max_step1 UNION ALL SELECT score_soft_max_step1."KEY" AS "KEY", 3 AS class, score_soft_max_step1."exp_Score_3" AS "exp_Score" 
FROM score_soft_max_step1) AS soft_scu), 
score_soft_max AS 
(SELECT score_soft_max_step1."KEY" AS "KEY", score_soft_max_step1."Score_0" AS "Score_0", score_soft_max_step1."exp_Score_0" AS "exp_Score_0", score_soft_max_step1."Score_1" AS "Score_1", score_soft_max_step1."exp_Score_1" AS "exp_Score_1", score_soft_max_step1."Score_2" AS "Score_2", score_soft_max_step1."exp_Score_2" AS "exp_Score_2", score_soft_max_step1."Score_3" AS "Score_3", score_soft_max_step1."exp_Score_3" AS "exp_Score_3", sum_exp_t."KEY_sum" AS "KEY_sum", sum_exp_t."sum_ExpScore" AS "sum_ExpScore" 
FROM score_soft_max_step1 LEFT OUTER JOIN (SELECT score_class_union_soft."KEY" AS "KEY_sum", sum(score_class_union_soft."exp_Score") AS "sum_ExpScore" 
FROM score_class_union_soft GROUP BY score_class_union_soft."KEY") AS sum_exp_t ON score_soft_max_step1."KEY" = sum_exp_t."KEY_sum"), 
layer_softmax AS 
(SELECT score_soft_max."KEY" AS "KEY", score_soft_max."exp_Score_0" / score_soft_max."sum_ExpScore" AS output_1, score_soft_max."exp_Score_1" / score_soft_max."sum_ExpScore" AS output_2, score_soft_max."exp_Score_2" / score_soft_max."sum_ExpScore" AS output_3, score_soft_max."exp_Score_3" / score_soft_max."sum_ExpScore" AS output_4 
FROM score_soft_max), 
orig_cte AS 
(SELECT layer_softmax."KEY" AS "KEY", CAST(NULL AS DOUBLE) AS "Score_0", CAST(NULL AS DOUBLE) AS "Score_1", CAST(NULL AS DOUBLE) AS "Score_2", CAST(NULL AS DOUBLE) AS "Score_3", layer_softmax.output_1 AS "Proba_0", layer_softmax.output_2 AS "Proba_1", layer_softmax.output_3 AS "Proba_2", layer_softmax.output_4 AS "Proba_3", CAST(NULL AS DOUBLE) AS "LogProba_0", CAST(NULL AS DOUBLE) AS "LogProba_1", CAST(NULL AS DOUBLE) AS "LogProba_2", CAST(NULL AS DOUBLE) AS "LogProba_3", CAST(NULL AS BIGINT) AS "Decision", CAST(NULL AS DOUBLE) AS "DecisionProba" 
FROM layer_softmax), 
score_class_union AS 
(SELECT scu."KEY_u" AS "KEY_u", scu.class AS class, scu."LogProba" AS "LogProba", scu."Proba" AS "Proba", scu."Score" AS "Score" 
FROM (SELECT orig_cte."KEY" AS "KEY_u", 0 AS class, orig_cte."LogProba_0" AS "LogProba", orig_cte."Proba_0" AS "Proba", orig_cte."Score_0" AS "Score" 
FROM orig_cte UNION ALL SELECT orig_cte."KEY" AS "KEY_u", 1 AS class, orig_cte."LogProba_1" AS "LogProba", orig_cte."Proba_1" AS "Proba", orig_cte."Score_1" AS "Score" 
FROM orig_cte UNION ALL SELECT orig_cte."KEY" AS "KEY_u", 2 AS class, orig_cte."LogProba_2" AS "LogProba", orig_cte."Proba_2" AS "Proba", orig_cte."Score_2" AS "Score" 
FROM orig_cte UNION ALL SELECT orig_cte."KEY" AS "KEY_u", 3 AS class, orig_cte."LogProba_3" AS "LogProba", orig_cte."Proba_3" AS "Proba", orig_cte."Score_3" AS "Score" 
FROM orig_cte) AS scu), 
score_max AS 
(SELECT orig_cte."KEY" AS "KEY", orig_cte."Score_0" AS "Score_0", orig_cte."Score_1" AS "Score_1", orig_cte."Score_2" AS "Score_2", orig_cte."Score_3" AS "Score_3", orig_cte."Proba_0" AS "Proba_0", orig_cte."Proba_1" AS "Proba_1", orig_cte."Proba_2" AS "Proba_2", orig_cte."Proba_3" AS "Proba_3", orig_cte."LogProba_0" AS "LogProba_0", orig_cte."LogProba_1" AS "LogProba_1", orig_cte."LogProba_2" AS "LogProba_2", orig_cte."LogProba_3" AS "LogProba_3", orig_cte."Decision" AS "Decision", orig_cte."DecisionProba" AS "DecisionProba", max_select."KEY_m" AS "KEY_m", max_select."max_Proba" AS "max_Proba" 
FROM orig_cte LEFT OUTER JOIN (SELECT score_class_union."KEY_u" AS "KEY_m", max(score_class_union."Proba") AS "max_Proba" 
FROM score_class_union GROUP BY score_class_union."KEY_u") AS max_select ON orig_cte."KEY" = max_select."KEY_m"), 
union_with_max AS 
(SELECT score_class_union."KEY_u" AS "KEY_u", score_class_union.class AS class, score_class_union."LogProba" AS "LogProba", score_class_union."Proba" AS "Proba", score_class_union."Score" AS "Score", score_max."KEY" AS "KEY", score_max."Score_0" AS "Score_0", score_max."Score_1" AS "Score_1", score_max."Score_2" AS "Score_2", score_max."Score_3" AS "Score_3", score_max."Proba_0" AS "Proba_0", score_max."Proba_1" AS "Proba_1", score_max."Proba_2" AS "Proba_2", score_max."Proba_3" AS "Proba_3", score_max."LogProba_0" AS "LogProba_0", score_max."LogProba_1" AS "LogProba_1", score_max."LogProba_2" AS "LogProba_2", score_max."LogProba_3" AS "LogProba_3", score_max."Decision" AS "Decision", score_max."DecisionProba" AS "DecisionProba", score_max."KEY_m" AS "KEY_m", score_max."max_Proba" AS "max_Proba" 
FROM score_class_union LEFT OUTER JOIN score_max ON score_class_union."KEY_u" = score_max."KEY"), 
arg_max_cte AS 
(SELECT score_max."KEY" AS "KEY", score_max."Score_0" AS "Score_0", score_max."Score_1" AS "Score_1", score_max."Score_2" AS "Score_2", score_max."Score_3" AS "Score_3", score_max."Proba_0" AS "Proba_0", score_max."Proba_1" AS "Proba_1", score_max."Proba_2" AS "Proba_2", score_max."Proba_3" AS "Proba_3", score_max."LogProba_0" AS "LogProba_0", score_max."LogProba_1" AS "LogProba_1", score_max."LogProba_2" AS "LogProba_2", score_max."LogProba_3" AS "LogProba_3", score_max."Decision" AS "Decision", score_max."DecisionProba" AS "DecisionProba", score_max."KEY_m" AS "KEY_m", score_max."max_Proba" AS "max_Proba", "arg_max_t_Proba"."KEY_Proba" AS "KEY_Proba", "arg_max_t_Proba"."arg_max_Proba" AS "arg_max_Proba" 
FROM score_max LEFT OUTER JOIN (SELECT union_with_max."KEY" AS "KEY_Proba", max(union_with_max.class) AS "arg_max_Proba" 
FROM union_with_max 
WHERE union_with_max."max_Proba" <= union_with_max."Proba" GROUP BY union_with_max."KEY") AS "arg_max_t_Proba" ON score_max."KEY" = "arg_max_t_Proba"."KEY_Proba")
 SELECT arg_max_cte."KEY" AS "KEY", arg_max_cte."Score_0" AS "Score_0", arg_max_cte."Score_1" AS "Score_1", arg_max_cte."Score_2" AS "Score_2", arg_max_cte."Score_3" AS "Score_3", arg_max_cte."Proba_0" AS "Proba_0", arg_max_cte."Proba_1" AS "Proba_1", arg_max_cte."Proba_2" AS "Proba_2", arg_max_cte."Proba_3" AS "Proba_3", log(CASE WHEN (arg_max_cte."Proba_0" IS NULL OR arg_max_cte."Proba_0" > 1e-100) THEN arg_max_cte."Proba_0" ELSE 1e-100 END) AS "LogProba_0", log(CASE WHEN (arg_max_cte."Proba_1" IS NULL OR arg_max_cte."Proba_1" > 1e-100) THEN arg_max_cte."Proba_1" ELSE 1e-100 END) AS "LogProba_1", log(CASE WHEN (arg_max_cte."Proba_2" IS NULL OR arg_max_cte."Proba_2" > 1e-100) THEN arg_max_cte."Proba_2" ELSE 1e-100 END) AS "LogProba_2", log(CASE WHEN (arg_max_cte."Proba_3" IS NULL OR arg_max_cte."Proba_3" > 1e-100) THEN arg_max_cte."Proba_3" ELSE 1e-100 END) AS "LogProba_3", arg_max_cte."arg_max_Proba" AS "Decision", arg_max_cte."max_Proba" AS "DecisionProba" 
FROM arg_max_cte