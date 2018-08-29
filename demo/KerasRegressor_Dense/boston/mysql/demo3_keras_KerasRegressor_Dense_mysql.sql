-- This SQL code was generated by sklearn2sql (development version).
-- Copyright 2018

-- Model : KerasRegressor_Dense
-- Dataset : boston
-- Database : mysql


-- This SQL code can contain one or more statements, to be executed in the order they appear in this file.



-- Model deployment code

WITH keras_input AS 
(SELECT `ADS`.`KEY` AS `KEY`, `ADS`.`Feature_0` AS `Feature_0`, `ADS`.`Feature_1` AS `Feature_1`, `ADS`.`Feature_2` AS `Feature_2`, `ADS`.`Feature_3` AS `Feature_3`, `ADS`.`Feature_4` AS `Feature_4`, `ADS`.`Feature_5` AS `Feature_5`, `ADS`.`Feature_6` AS `Feature_6`, `ADS`.`Feature_7` AS `Feature_7`, `ADS`.`Feature_8` AS `Feature_8`, `ADS`.`Feature_9` AS `Feature_9`, `ADS`.`Feature_10` AS `Feature_10`, `ADS`.`Feature_11` AS `Feature_11`, `ADS`.`Feature_12` AS `Feature_12` 
FROM boston AS `ADS`), 
keras_input_1 AS 
(SELECT keras_input.`KEY` AS `KEY`, keras_input.`Feature_0` AS `Feature_0`, keras_input.`Feature_1` AS `Feature_1`, keras_input.`Feature_2` AS `Feature_2`, keras_input.`Feature_3` AS `Feature_3`, keras_input.`Feature_4` AS `Feature_4`, keras_input.`Feature_5` AS `Feature_5`, keras_input.`Feature_6` AS `Feature_6`, keras_input.`Feature_7` AS `Feature_7`, keras_input.`Feature_8` AS `Feature_8`, keras_input.`Feature_9` AS `Feature_9`, keras_input.`Feature_10` AS `Feature_10`, keras_input.`Feature_11` AS `Feature_11`, keras_input.`Feature_12` AS `Feature_12` 
FROM keras_input), 
layer_dense_1 AS 
(SELECT keras_input_1.`KEY` AS `KEY`, 0.2845376133918762 + 0.05955107510089874 * keras_input_1.`Feature_0` + -0.012716525234282017 * keras_input_1.`Feature_1` + -0.0282120443880558 * keras_input_1.`Feature_2` + 0.296810120344162 * keras_input_1.`Feature_3` + 0.7616167664527893 * keras_input_1.`Feature_4` + 0.5312532186508179 * keras_input_1.`Feature_5` + 0.09234975278377533 * keras_input_1.`Feature_6` + -0.02212083898484707 * keras_input_1.`Feature_7` + 0.1900424063205719 * keras_input_1.`Feature_8` + -0.3167722225189209 * keras_input_1.`Feature_9` + 0.1888277530670166 * keras_input_1.`Feature_10` + 0.22367112338542938 * keras_input_1.`Feature_11` + -0.19093573093414307 * keras_input_1.`Feature_12` AS output_1, 0.2788756489753723 + -0.06381690502166748 * keras_input_1.`Feature_0` + 0.019700001925230026 * keras_input_1.`Feature_1` + -0.6933981776237488 * keras_input_1.`Feature_2` + 0.8735244870185852 * keras_input_1.`Feature_3` + 0.15080317854881287 * keras_input_1.`Feature_4` + 1.0456267595291138 * keras_input_1.`Feature_5` + 0.6517165899276733 * keras_input_1.`Feature_6` + -0.34780141711235046 * keras_input_1.`Feature_7` + -0.46109211444854736 * keras_input_1.`Feature_8` + -0.5044503211975098 * keras_input_1.`Feature_9` + 0.5861343741416931 * keras_input_1.`Feature_10` + 0.5433911085128784 * keras_input_1.`Feature_11` + -0.5125572085380554 * keras_input_1.`Feature_12` AS output_2, 0.26696717739105225 + -0.46383193135261536 * keras_input_1.`Feature_0` + 0.007466757670044899 * keras_input_1.`Feature_1` + 0.283869206905365 * keras_input_1.`Feature_2` + 0.6944233179092407 * keras_input_1.`Feature_3` + -0.10790511965751648 * keras_input_1.`Feature_4` + 0.9667580723762512 * keras_input_1.`Feature_5` + -0.3542602062225342 * keras_input_1.`Feature_6` + -0.03315440192818642 * keras_input_1.`Feature_7` + 0.19780732691287994 * keras_input_1.`Feature_8` + 0.5135297179222107 * keras_input_1.`Feature_9` + -0.2728517949581146 * keras_input_1.`Feature_10` + -0.12412841618061066 * keras_input_1.`Feature_11` + -0.3203667998313904 * keras_input_1.`Feature_12` AS output_3, 0.2769593596458435 + 0.5281705260276794 * keras_input_1.`Feature_0` + 0.35346558690071106 * keras_input_1.`Feature_1` + -0.003923144191503525 * keras_input_1.`Feature_2` + 0.20201577246189117 * keras_input_1.`Feature_3` + -0.2633218765258789 * keras_input_1.`Feature_4` + 0.728621780872345 * keras_input_1.`Feature_5` + 0.4924734830856323 * keras_input_1.`Feature_6` + 0.15651986002922058 * keras_input_1.`Feature_7` + 0.32228508591651917 * keras_input_1.`Feature_8` + -0.5082405209541321 * keras_input_1.`Feature_9` + -0.11945313215255737 * keras_input_1.`Feature_10` + -0.2318643033504486 * keras_input_1.`Feature_11` + 0.23899559676647186 * keras_input_1.`Feature_12` AS output_4 
FROM keras_input_1), 
layer_dense_1_1 AS 
(SELECT layer_dense_1.`KEY` AS `KEY`, layer_dense_1.output_1 AS output_1, layer_dense_1.output_2 AS output_2, layer_dense_1.output_3 AS output_3, layer_dense_1.output_4 AS output_4 
FROM layer_dense_1), 
layer_dense_2 AS 
(SELECT layer_dense_1_1.`KEY` AS `KEY`, 0.27547451853752136 + 0.48430049419403076 * layer_dense_1_1.output_1 + 0.18099011480808258 * layer_dense_1_1.output_2 + 0.7431886792182922 * layer_dense_1_1.output_3 + 0.30752360820770264 * layer_dense_1_1.output_4 AS output_1 
FROM layer_dense_1_1)
 SELECT layer_dense_2.`KEY` AS `KEY`, layer_dense_2.output_1 AS `Estimator` 
FROM layer_dense_2