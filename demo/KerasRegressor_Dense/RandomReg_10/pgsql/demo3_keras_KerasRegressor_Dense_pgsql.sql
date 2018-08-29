-- This SQL code was generated by sklearn2sql (development version).
-- Copyright 2018

-- Model : KerasRegressor_Dense
-- Dataset : RandomReg_10
-- Database : pgsql


-- This SQL code can contain one or more statements, to be executed in the order they appear in this file.



-- Model deployment code

WITH keras_input AS 
(SELECT "ADS"."KEY" AS "KEY", "ADS"."Feature_0" AS "Feature_0", "ADS"."Feature_1" AS "Feature_1", "ADS"."Feature_2" AS "Feature_2", "ADS"."Feature_3" AS "Feature_3", "ADS"."Feature_4" AS "Feature_4", "ADS"."Feature_5" AS "Feature_5", "ADS"."Feature_6" AS "Feature_6", "ADS"."Feature_7" AS "Feature_7", "ADS"."Feature_8" AS "Feature_8", "ADS"."Feature_9" AS "Feature_9" 
FROM "RandomReg_10" AS "ADS"), 
keras_input_1 AS 
(SELECT keras_input."KEY" AS "KEY", keras_input."Feature_0" AS "Feature_0", keras_input."Feature_1" AS "Feature_1", keras_input."Feature_2" AS "Feature_2", keras_input."Feature_3" AS "Feature_3", keras_input."Feature_4" AS "Feature_4", keras_input."Feature_5" AS "Feature_5", keras_input."Feature_6" AS "Feature_6", keras_input."Feature_7" AS "Feature_7", keras_input."Feature_8" AS "Feature_8", keras_input."Feature_9" AS "Feature_9" 
FROM keras_input), 
layer_dense_1 AS 
(SELECT keras_input_1."KEY" AS "KEY", 0.1419800966978073 + 0.26204174757003784 * keras_input_1."Feature_0" + 0.3841208815574646 * keras_input_1."Feature_1" + 0.5712944865226746 * keras_input_1."Feature_2" + 0.1953662484884262 * keras_input_1."Feature_3" + 0.8147903680801392 * keras_input_1."Feature_4" + 0.804478645324707 * keras_input_1."Feature_5" + -0.31129446625709534 * keras_input_1."Feature_6" + 0.4828197956085205 * keras_input_1."Feature_7" + 0.6814486980438232 * keras_input_1."Feature_8" + 0.48069238662719727 * keras_input_1."Feature_9" AS output_1, -0.13600711524486542 + -0.011978679336607456 * keras_input_1."Feature_0" + -0.9193530678749084 * keras_input_1."Feature_1" + -0.11139402538537979 * keras_input_1."Feature_2" + -0.28527796268463135 * keras_input_1."Feature_3" + -0.902597188949585 * keras_input_1."Feature_4" + -0.12258228659629822 * keras_input_1."Feature_5" + -0.2587432861328125 * keras_input_1."Feature_6" + -0.09153126925230026 * keras_input_1."Feature_7" + -0.5104985237121582 * keras_input_1."Feature_8" + -0.10958726704120636 * keras_input_1."Feature_9" AS output_2, -0.13009032607078552 + -0.3267833888530731 * keras_input_1."Feature_0" + -0.151468425989151 * keras_input_1."Feature_1" + -0.06234147027134895 * keras_input_1."Feature_2" + 0.2239382117986679 * keras_input_1."Feature_3" + -0.9354475736618042 * keras_input_1."Feature_4" + -0.5818146467208862 * keras_input_1."Feature_5" + 0.2172224223613739 * keras_input_1."Feature_6" + -0.262151837348938 * keras_input_1."Feature_7" + -0.7499102354049683 * keras_input_1."Feature_8" + 0.05675714462995529 * keras_input_1."Feature_9" AS output_3, 0.13802111148834229 + 0.47166454792022705 * keras_input_1."Feature_0" + 0.43129482865333557 * keras_input_1."Feature_1" + -0.02089398354291916 * keras_input_1."Feature_2" + 0.20441116392612457 * keras_input_1."Feature_3" + 0.5140386819839478 * keras_input_1."Feature_4" + 0.6498228311538696 * keras_input_1."Feature_5" + 0.6946321129798889 * keras_input_1."Feature_6" + 0.6109402775764465 * keras_input_1."Feature_7" + 0.5873400568962097 * keras_input_1."Feature_8" + -0.08054489642381668 * keras_input_1."Feature_9" AS output_4 
FROM keras_input_1), 
layer_dense_1_1 AS 
(SELECT layer_dense_1."KEY" AS "KEY", layer_dense_1.output_1 AS output_1, layer_dense_1.output_2 AS output_2, layer_dense_1.output_3 AS output_3, layer_dense_1.output_4 AS output_4 
FROM layer_dense_1), 
layer_dense_2 AS 
(SELECT layer_dense_1_1."KEY" AS "KEY", 0.12295302748680115 + 1.1123507022857666 * layer_dense_1_1.output_1 + -1.3136461973190308 * layer_dense_1_1.output_2 + -1.295338749885559 * layer_dense_1_1.output_3 + 1.042354702949524 * layer_dense_1_1.output_4 AS output_1 
FROM layer_dense_1_1)
 SELECT layer_dense_2."KEY" AS "KEY", layer_dense_2.output_1 AS "Estimator" 
FROM layer_dense_2