-- This SQL code was generated by sklearn2sql (development version).
-- Copyright 2018

-- Model : KerasRegressor_Dense
-- Dataset : diabetes
-- Database : oracle


-- This SQL code can contain one or more statements, to be executed in the order they appear in this file.



-- Model deployment code

WITH keras_input AS 
(SELECT "ADS"."KEY" AS "KEY", "ADS"."Feature_0" AS "Feature_0", "ADS"."Feature_1" AS "Feature_1", "ADS"."Feature_2" AS "Feature_2", "ADS"."Feature_3" AS "Feature_3", "ADS"."Feature_4" AS "Feature_4", "ADS"."Feature_5" AS "Feature_5", "ADS"."Feature_6" AS "Feature_6", "ADS"."Feature_7" AS "Feature_7", "ADS"."Feature_8" AS "Feature_8", "ADS"."Feature_9" AS "Feature_9" 
FROM "DIABETES" "ADS"), 
keras_input_1 AS 
(SELECT keras_input."KEY" AS "KEY", keras_input."Feature_0" AS "Feature_0", keras_input."Feature_1" AS "Feature_1", keras_input."Feature_2" AS "Feature_2", keras_input."Feature_3" AS "Feature_3", keras_input."Feature_4" AS "Feature_4", keras_input."Feature_5" AS "Feature_5", keras_input."Feature_6" AS "Feature_6", keras_input."Feature_7" AS "Feature_7", keras_input."Feature_8" AS "Feature_8", keras_input."Feature_9" AS "Feature_9" 
FROM keras_input), 
layer_dense_1 AS 
(SELECT keras_input_1."KEY" AS "KEY", 4.411129474639893 + 0.5860896110534668 * keras_input_1."Feature_0" + 0.039443183690309525 * keras_input_1."Feature_1" + 1.9839533567428589 * keras_input_1."Feature_2" + 1.2052780389785767 * keras_input_1."Feature_3" + 1.0104222297668457 * keras_input_1."Feature_4" + 0.7351000905036926 * keras_input_1."Feature_5" + -1.569513201713562 * keras_input_1."Feature_6" + 1.0677210092544556 * keras_input_1."Feature_7" + 1.241227149963379 * keras_input_1."Feature_8" + 0.8844428062438965 * keras_input_1."Feature_9" AS output_1, -4.284599781036377 + -0.34022992849349976 * keras_input_1."Feature_0" + 0.17320270836353302 * keras_input_1."Feature_1" + -1.9666181802749634 * keras_input_1."Feature_2" + -0.8598591089248657 * keras_input_1."Feature_3" + -0.07711999118328094 * keras_input_1."Feature_4" + 0.04986407235264778 * keras_input_1."Feature_5" + 1.9476327896118164 * keras_input_1."Feature_6" + -1.6659750938415527 * keras_input_1."Feature_7" + -1.744236707687378 * keras_input_1."Feature_8" + -0.6302632689476013 * keras_input_1."Feature_9" AS output_2, 4.403477191925049 + 0.7371311783790588 * keras_input_1."Feature_0" + 0.742042064666748 * keras_input_1."Feature_1" + 1.836326003074646 * keras_input_1."Feature_2" + 0.6478397250175476 * keras_input_1."Feature_3" + 1.254043459892273 * keras_input_1."Feature_4" + -0.04470057412981987 * keras_input_1."Feature_5" + -1.1816242933273315 * keras_input_1."Feature_6" + 1.5724624395370483 * keras_input_1."Feature_7" + 1.3436251878738403 * keras_input_1."Feature_8" + 0.6317874789237976 * keras_input_1."Feature_9" AS output_3, -4.614416599273682 + -1.0036680698394775 * keras_input_1."Feature_0" + -0.11688989400863647 * keras_input_1."Feature_1" + -1.5753142833709717 * keras_input_1."Feature_2" + -0.5898033380508423 * keras_input_1."Feature_3" + -0.4063929617404938 * keras_input_1."Feature_4" + -0.7514408826828003 * keras_input_1."Feature_5" + 1.4771369695663452 * keras_input_1."Feature_6" + -1.3612254858016968 * keras_input_1."Feature_7" + -2.1067185401916504 * keras_input_1."Feature_8" + -0.8255149722099304 * keras_input_1."Feature_9" AS output_4 
FROM keras_input_1), 
layer_dense_1_1 AS 
(SELECT layer_dense_1."KEY" AS "KEY", layer_dense_1.output_1 AS output_1, layer_dense_1.output_2 AS output_2, layer_dense_1.output_3 AS output_3, layer_dense_1.output_4 AS output_4 
FROM layer_dense_1), 
layer_dense_2 AS 
(SELECT layer_dense_1_1."KEY" AS "KEY", 3.239513635635376 + 4.815432071685791 * layer_dense_1_1.output_1 + -4.972939968109131 * layer_dense_1_1.output_2 + 4.863869667053223 * layer_dense_1_1.output_3 + -4.6096978187561035 * layer_dense_1_1.output_4 AS output_1 
FROM layer_dense_1_1)
 SELECT layer_dense_2."KEY" AS "KEY", layer_dense_2.output_1 AS "Estimator" 
FROM layer_dense_2