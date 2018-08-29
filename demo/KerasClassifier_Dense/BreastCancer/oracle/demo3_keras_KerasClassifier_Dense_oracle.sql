-- This SQL code was generated by sklearn2sql (development version).
-- Copyright 2018

-- Model : KerasClassifier_Dense
-- Dataset : BreastCancer
-- Database : oracle


-- This SQL code can contain one or more statements, to be executed in the order they appear in this file.



-- Model deployment code

WITH keras_input AS 
(SELECT "ADS"."KEY" AS "KEY", "ADS"."Feature_0" AS "Feature_0", "ADS"."Feature_1" AS "Feature_1", "ADS"."Feature_2" AS "Feature_2", "ADS"."Feature_3" AS "Feature_3", "ADS"."Feature_4" AS "Feature_4", "ADS"."Feature_5" AS "Feature_5", "ADS"."Feature_6" AS "Feature_6", "ADS"."Feature_7" AS "Feature_7", "ADS"."Feature_8" AS "Feature_8", "ADS"."Feature_9" AS "Feature_9", "ADS"."Feature_10" AS "Feature_10", "ADS"."Feature_11" AS "Feature_11", "ADS"."Feature_12" AS "Feature_12", "ADS"."Feature_13" AS "Feature_13", "ADS"."Feature_14" AS "Feature_14", "ADS"."Feature_15" AS "Feature_15", "ADS"."Feature_16" AS "Feature_16", "ADS"."Feature_17" AS "Feature_17", "ADS"."Feature_18" AS "Feature_18", "ADS"."Feature_19" AS "Feature_19", "ADS"."Feature_20" AS "Feature_20", "ADS"."Feature_21" AS "Feature_21", "ADS"."Feature_22" AS "Feature_22", "ADS"."Feature_23" AS "Feature_23", "ADS"."Feature_24" AS "Feature_24", "ADS"."Feature_25" AS "Feature_25", "ADS"."Feature_26" AS "Feature_26", "ADS"."Feature_27" AS "Feature_27", "ADS"."Feature_28" AS "Feature_28", "ADS"."Feature_29" AS "Feature_29" 
FROM "BREASTCANCER" "ADS"), 
keras_input_1 AS 
(SELECT keras_input."KEY" AS "KEY", keras_input."Feature_0" AS "Feature_0", keras_input."Feature_1" AS "Feature_1", keras_input."Feature_2" AS "Feature_2", keras_input."Feature_3" AS "Feature_3", keras_input."Feature_4" AS "Feature_4", keras_input."Feature_5" AS "Feature_5", keras_input."Feature_6" AS "Feature_6", keras_input."Feature_7" AS "Feature_7", keras_input."Feature_8" AS "Feature_8", keras_input."Feature_9" AS "Feature_9", keras_input."Feature_10" AS "Feature_10", keras_input."Feature_11" AS "Feature_11", keras_input."Feature_12" AS "Feature_12", keras_input."Feature_13" AS "Feature_13", keras_input."Feature_14" AS "Feature_14", keras_input."Feature_15" AS "Feature_15", keras_input."Feature_16" AS "Feature_16", keras_input."Feature_17" AS "Feature_17", keras_input."Feature_18" AS "Feature_18", keras_input."Feature_19" AS "Feature_19", keras_input."Feature_20" AS "Feature_20", keras_input."Feature_21" AS "Feature_21", keras_input."Feature_22" AS "Feature_22", keras_input."Feature_23" AS "Feature_23", keras_input."Feature_24" AS "Feature_24", keras_input."Feature_25" AS "Feature_25", keras_input."Feature_26" AS "Feature_26", keras_input."Feature_27" AS "Feature_27", keras_input."Feature_28" AS "Feature_28", keras_input."Feature_29" AS "Feature_29" 
FROM keras_input), 
layer_dense_1 AS 
(SELECT keras_input_1."KEY" AS "KEY", 0.0 + 0.015125930309295654 * keras_input_1."Feature_0" + 0.020204931497573853 * keras_input_1."Feature_1" + 0.03913608193397522 * keras_input_1."Feature_2" + -0.21991978585720062 * keras_input_1."Feature_3" + -0.11366134881973267 * keras_input_1."Feature_4" + -0.25943246483802795 * keras_input_1."Feature_5" + 0.19885823130607605 * keras_input_1."Feature_6" + 0.3892751634120941 * keras_input_1."Feature_7" + -0.10722294449806213 * keras_input_1."Feature_8" + -0.034910231828689575 * keras_input_1."Feature_9" + -0.08615043759346008 * keras_input_1."Feature_10" + 0.02920505404472351 * keras_input_1."Feature_11" + 0.36923810839653015 * keras_input_1."Feature_12" + 0.37817832827568054 * keras_input_1."Feature_13" + -0.08335089683532715 * keras_input_1."Feature_14" + 0.28128835558891296 * keras_input_1."Feature_15" + -0.2514112889766693 * keras_input_1."Feature_16" + -0.15092572569847107 * keras_input_1."Feature_17" + 0.24039623141288757 * keras_input_1."Feature_18" + -0.10864725708961487 * keras_input_1."Feature_19" + -0.21136046946048737 * keras_input_1."Feature_20" + 0.12973114848136902 * keras_input_1."Feature_21" + -0.3802115023136139 * keras_input_1."Feature_22" + 0.021149903535842896 * keras_input_1."Feature_23" + -0.3663949966430664 * keras_input_1."Feature_24" + 0.3774949610233307 * keras_input_1."Feature_25" + -0.14382889866828918 * keras_input_1."Feature_26" + 0.3217799961566925 * keras_input_1."Feature_27" + 0.06628203392028809 * keras_input_1."Feature_28" + -0.18718916177749634 * keras_input_1."Feature_29" AS output_1, -0.07065892964601517 + 0.18375235795974731 * keras_input_1."Feature_0" + -0.204927459359169 * keras_input_1."Feature_1" + -0.0232579093426466 * keras_input_1."Feature_2" + -0.17801322042942047 * keras_input_1."Feature_3" + 0.3557361960411072 * keras_input_1."Feature_4" + -0.2158786505460739 * keras_input_1."Feature_5" + 0.36475592851638794 * keras_input_1."Feature_6" + 0.18826250731945038 * keras_input_1."Feature_7" + 0.022830113768577576 * keras_input_1."Feature_8" + 0.018666859716176987 * keras_input_1."Feature_9" + -0.3180198669433594 * keras_input_1."Feature_10" + -0.4322599172592163 * keras_input_1."Feature_11" + -0.18717600405216217 * keras_input_1."Feature_12" + -0.14218655228614807 * keras_input_1."Feature_13" + 0.30992287397384644 * keras_input_1."Feature_14" + 0.3151003122329712 * keras_input_1."Feature_15" + 0.13063403964042664 * keras_input_1."Feature_16" + 0.0668569952249527 * keras_input_1."Feature_17" + -0.18624764680862427 * keras_input_1."Feature_18" + -0.14461250603199005 * keras_input_1."Feature_19" + -0.005253639072179794 * keras_input_1."Feature_20" + 0.2722278833389282 * keras_input_1."Feature_21" + -0.34604698419570923 * keras_input_1."Feature_22" + 0.1424807757139206 * keras_input_1."Feature_23" + 0.3532751500606537 * keras_input_1."Feature_24" + -0.09304028749465942 * keras_input_1."Feature_25" + 0.3582630157470703 * keras_input_1."Feature_26" + -0.3610444664955139 * keras_input_1."Feature_27" + 0.031667560338974 * keras_input_1."Feature_28" + -0.18325494229793549 * keras_input_1."Feature_29" AS output_2, 0.0 + -0.07862752676010132 * keras_input_1."Feature_0" + 0.3745223581790924 * keras_input_1."Feature_1" + 0.21952655911445618 * keras_input_1."Feature_2" + 0.019506752490997314 * keras_input_1."Feature_3" + 0.1677853763103485 * keras_input_1."Feature_4" + -0.01471930742263794 * keras_input_1."Feature_5" + 0.2985479533672333 * keras_input_1."Feature_6" + -0.33771055936813354 * keras_input_1."Feature_7" + -0.3016822934150696 * keras_input_1."Feature_8" + -0.15584257245063782 * keras_input_1."Feature_9" + -0.4001525044441223 * keras_input_1."Feature_10" + -0.03105872869491577 * keras_input_1."Feature_11" + -0.18131351470947266 * keras_input_1."Feature_12" + 0.3992839753627777 * keras_input_1."Feature_13" + -0.3026808500289917 * keras_input_1."Feature_14" + 0.13396862149238586 * keras_input_1."Feature_15" + 0.34278425574302673 * keras_input_1."Feature_16" + 0.12653598189353943 * keras_input_1."Feature_17" + -0.22853118181228638 * keras_input_1."Feature_18" + -0.08934050798416138 * keras_input_1."Feature_19" + 0.07318577170372009 * keras_input_1."Feature_20" + 0.24496176838874817 * keras_input_1."Feature_21" + 0.3721710741519928 * keras_input_1."Feature_22" + 0.13056573271751404 * keras_input_1."Feature_23" + 0.1482396423816681 * keras_input_1."Feature_24" + -0.009328722953796387 * keras_input_1."Feature_25" + 0.07787007093429565 * keras_input_1."Feature_26" + 0.19337621331214905 * keras_input_1."Feature_27" + 0.3551947772502899 * keras_input_1."Feature_28" + -0.22355924546718597 * keras_input_1."Feature_29" AS output_3, 0.0 + -0.2811186909675598 * keras_input_1."Feature_0" + 0.3642089068889618 * keras_input_1."Feature_1" + 0.2529478967189789 * keras_input_1."Feature_2" + -0.1407669186592102 * keras_input_1."Feature_3" + -0.12573614716529846 * keras_input_1."Feature_4" + 0.2634001076221466 * keras_input_1."Feature_5" + 0.4021255671977997 * keras_input_1."Feature_6" + -0.39983871579170227 * keras_input_1."Feature_7" + -0.10261967778205872 * keras_input_1."Feature_8" + 0.33347955346107483 * keras_input_1."Feature_9" + 0.008007466793060303 * keras_input_1."Feature_10" + -0.026369929313659668 * keras_input_1."Feature_11" + -0.25131094455718994 * keras_input_1."Feature_12" + -0.08449828624725342 * keras_input_1."Feature_13" + 0.07926732301712036 * keras_input_1."Feature_14" + 0.40458592772483826 * keras_input_1."Feature_15" + -0.03406339883804321 * keras_input_1."Feature_16" + 0.013976961374282837 * keras_input_1."Feature_17" + 0.10183271765708923 * keras_input_1."Feature_18" + -0.23613601922988892 * keras_input_1."Feature_19" + 0.08991608023643494 * keras_input_1."Feature_20" + 0.13576212525367737 * keras_input_1."Feature_21" + 0.2490963637828827 * keras_input_1."Feature_22" + -0.36495766043663025 * keras_input_1."Feature_23" + -0.13210514187812805 * keras_input_1."Feature_24" + 0.10800644755363464 * keras_input_1."Feature_25" + -0.2516134977340698 * keras_input_1."Feature_26" + -0.16417965292930603 * keras_input_1."Feature_27" + -0.3737332224845886 * keras_input_1."Feature_28" + 0.36429932713508606 * keras_input_1."Feature_29" AS output_4 
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
(SELECT activation_relu_1_1."KEY" AS "KEY", -0.3901517391204834 + 0.234405517578125 * activation_relu_1_1.output_1 + -0.24682222306728363 * activation_relu_1_1.output_2 + -0.013787031173706055 * activation_relu_1_1.output_3 + 0.4626593589782715 * activation_relu_1_1.output_4 AS output_1, 0.390151709318161 + 0.6107501983642578 * activation_relu_1_1.output_1 + -0.8403566479682922 * activation_relu_1_1.output_2 + 0.9594423770904541 * activation_relu_1_1.output_3 + 0.4524827003479004 * activation_relu_1_1.output_4 AS output_2 
FROM activation_relu_1_1), 
layer_dense_2_1 AS 
(SELECT layer_dense_2."KEY" AS "KEY", layer_dense_2.output_1 AS output_1, layer_dense_2.output_2 AS output_2 
FROM layer_dense_2), 
score_soft_max_step1 AS 
(SELECT layer_dense_2_1."KEY" AS "KEY", layer_dense_2_1.output_1 AS "Score_0", exp(least(100.0, greatest(-100.0, layer_dense_2_1.output_1))) AS "exp_Score_0", layer_dense_2_1.output_2 AS "Score_1", exp(least(100.0, greatest(-100.0, layer_dense_2_1.output_2))) AS "exp_Score_1" 
FROM layer_dense_2_1), 
score_class_union_soft AS 
(SELECT soft_scu."KEY" AS "KEY", soft_scu.class AS class, soft_scu."exp_Score" AS "exp_Score" 
FROM (SELECT score_soft_max_step1."KEY" AS "KEY", 0 AS class, score_soft_max_step1."exp_Score_0" AS "exp_Score" 
FROM score_soft_max_step1 UNION ALL SELECT score_soft_max_step1."KEY" AS "KEY", 1 AS class, score_soft_max_step1."exp_Score_1" AS "exp_Score" 
FROM score_soft_max_step1) soft_scu), 
score_soft_max AS 
(SELECT score_soft_max_step1."KEY" AS "KEY", score_soft_max_step1."Score_0" AS "Score_0", score_soft_max_step1."exp_Score_0" AS "exp_Score_0", score_soft_max_step1."Score_1" AS "Score_1", score_soft_max_step1."exp_Score_1" AS "exp_Score_1", sum_exp_t."KEY_sum" AS "KEY_sum", sum_exp_t."sum_ExpScore" AS "sum_ExpScore" 
FROM score_soft_max_step1 LEFT OUTER JOIN (SELECT score_class_union_soft."KEY" AS "KEY_sum", sum(score_class_union_soft."exp_Score") AS "sum_ExpScore" 
FROM score_class_union_soft GROUP BY score_class_union_soft."KEY") sum_exp_t ON score_soft_max_step1."KEY" = sum_exp_t."KEY_sum"), 
layer_softmax AS 
(SELECT score_soft_max."KEY" AS "KEY", score_soft_max."exp_Score_0" / score_soft_max."sum_ExpScore" AS output_1, score_soft_max."exp_Score_1" / score_soft_max."sum_ExpScore" AS output_2 
FROM score_soft_max), 
orig_cte AS 
(SELECT layer_softmax."KEY" AS "KEY", CAST(NULL AS BINARY_DOUBLE) AS "Score_0", CAST(NULL AS BINARY_DOUBLE) AS "Score_1", layer_softmax.output_1 AS "Proba_0", layer_softmax.output_2 AS "Proba_1", CAST(NULL AS BINARY_DOUBLE) AS "LogProba_0", CAST(NULL AS BINARY_DOUBLE) AS "LogProba_1", CAST(NULL AS NUMBER(19)) AS "Decision", CAST(NULL AS BINARY_DOUBLE) AS "DecisionProba" 
FROM layer_softmax), 
score_class_union AS 
(SELECT scu."KEY_u" AS "KEY_u", scu.class AS class, scu."LogProba" AS "LogProba", scu."Proba" AS "Proba", scu."Score" AS "Score" 
FROM (SELECT orig_cte."KEY" AS "KEY_u", 0 AS class, orig_cte."LogProba_0" AS "LogProba", orig_cte."Proba_0" AS "Proba", orig_cte."Score_0" AS "Score" 
FROM orig_cte UNION ALL SELECT orig_cte."KEY" AS "KEY_u", 1 AS class, orig_cte."LogProba_1" AS "LogProba", orig_cte."Proba_1" AS "Proba", orig_cte."Score_1" AS "Score" 
FROM orig_cte) scu), 
score_max AS 
(SELECT orig_cte."KEY" AS "KEY", orig_cte."Score_0" AS "Score_0", orig_cte."Score_1" AS "Score_1", orig_cte."Proba_0" AS "Proba_0", orig_cte."Proba_1" AS "Proba_1", orig_cte."LogProba_0" AS "LogProba_0", orig_cte."LogProba_1" AS "LogProba_1", orig_cte."Decision" AS "Decision", orig_cte."DecisionProba" AS "DecisionProba", max_select."KEY_m" AS "KEY_m", max_select."max_Proba" AS "max_Proba" 
FROM orig_cte LEFT OUTER JOIN (SELECT score_class_union."KEY_u" AS "KEY_m", max(score_class_union."Proba") AS "max_Proba" 
FROM score_class_union GROUP BY score_class_union."KEY_u") max_select ON orig_cte."KEY" = max_select."KEY_m"), 
union_with_max AS 
(SELECT score_class_union."KEY_u" AS "KEY_u", score_class_union.class AS class, score_class_union."LogProba" AS "LogProba", score_class_union."Proba" AS "Proba", score_class_union."Score" AS "Score", score_max."KEY" AS "KEY", score_max."Score_0" AS "Score_0", score_max."Score_1" AS "Score_1", score_max."Proba_0" AS "Proba_0", score_max."Proba_1" AS "Proba_1", score_max."LogProba_0" AS "LogProba_0", score_max."LogProba_1" AS "LogProba_1", score_max."Decision" AS "Decision", score_max."DecisionProba" AS "DecisionProba", score_max."KEY_m" AS "KEY_m", score_max."max_Proba" AS "max_Proba" 
FROM score_class_union LEFT OUTER JOIN score_max ON score_class_union."KEY_u" = score_max."KEY"), 
arg_max_cte AS 
(SELECT score_max."KEY" AS "KEY", score_max."Score_0" AS "Score_0", score_max."Score_1" AS "Score_1", score_max."Proba_0" AS "Proba_0", score_max."Proba_1" AS "Proba_1", score_max."LogProba_0" AS "LogProba_0", score_max."LogProba_1" AS "LogProba_1", score_max."Decision" AS "Decision", score_max."DecisionProba" AS "DecisionProba", score_max."KEY_m" AS "KEY_m", score_max."max_Proba" AS "max_Proba", "arg_max_t_Proba"."KEY_Proba" AS "KEY_Proba", "arg_max_t_Proba"."arg_max_Proba" AS "arg_max_Proba" 
FROM score_max LEFT OUTER JOIN (SELECT union_with_max."KEY" AS "KEY_Proba", max(union_with_max.class) AS "arg_max_Proba" 
FROM union_with_max 
WHERE union_with_max."max_Proba" <= union_with_max."Proba" GROUP BY union_with_max."KEY") "arg_max_t_Proba" ON score_max."KEY" = "arg_max_t_Proba"."KEY_Proba")
 SELECT arg_max_cte."KEY" AS "KEY", arg_max_cte."Score_0" AS "Score_0", arg_max_cte."Score_1" AS "Score_1", arg_max_cte."Proba_0" AS "Proba_0", arg_max_cte."Proba_1" AS "Proba_1", CASE WHEN (arg_max_cte."Proba_0" IS NULL OR arg_max_cte."Proba_0" > 0.0) THEN ln(arg_max_cte."Proba_0") ELSE -BINARY_DOUBLE_INFINITY END AS "LogProba_0", CASE WHEN (arg_max_cte."Proba_1" IS NULL OR arg_max_cte."Proba_1" > 0.0) THEN ln(arg_max_cte."Proba_1") ELSE -BINARY_DOUBLE_INFINITY END AS "LogProba_1", arg_max_cte."arg_max_Proba" AS "Decision", arg_max_cte."max_Proba" AS "DecisionProba" 
FROM arg_max_cte