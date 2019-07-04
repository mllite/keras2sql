-- This SQL code was generated by sklearn2sql (development version).
-- Copyright 2018

-- Model : KerasClassifier_GRU
-- Dataset : BreastCancer
-- Database : mssql


-- This SQL code can contain one or more statements, to be executed in the order they appear in this file.



-- Model deployment code

WITH keras_input AS 
(SELECT [ADS].[KEY] AS [KEY], [ADS].[Feature_0] AS [Feature_0], [ADS].[Feature_1] AS [Feature_1], [ADS].[Feature_2] AS [Feature_2], [ADS].[Feature_3] AS [Feature_3], [ADS].[Feature_4] AS [Feature_4], [ADS].[Feature_5] AS [Feature_5], [ADS].[Feature_6] AS [Feature_6], [ADS].[Feature_7] AS [Feature_7], [ADS].[Feature_8] AS [Feature_8], [ADS].[Feature_9] AS [Feature_9], [ADS].[Feature_10] AS [Feature_10], [ADS].[Feature_11] AS [Feature_11], [ADS].[Feature_12] AS [Feature_12], [ADS].[Feature_13] AS [Feature_13], [ADS].[Feature_14] AS [Feature_14], [ADS].[Feature_15] AS [Feature_15], [ADS].[Feature_16] AS [Feature_16], [ADS].[Feature_17] AS [Feature_17], [ADS].[Feature_18] AS [Feature_18], [ADS].[Feature_19] AS [Feature_19], [ADS].[Feature_20] AS [Feature_20], [ADS].[Feature_21] AS [Feature_21], [ADS].[Feature_22] AS [Feature_22], [ADS].[Feature_23] AS [Feature_23], [ADS].[Feature_24] AS [Feature_24], [ADS].[Feature_25] AS [Feature_25], [ADS].[Feature_26] AS [Feature_26], [ADS].[Feature_27] AS [Feature_27], [ADS].[Feature_28] AS [Feature_28], [ADS].[Feature_29] AS [Feature_29] 
FROM [BreastCancer] AS [ADS]), 
keras_input_1 AS 
(SELECT keras_input.[KEY] AS [KEY], keras_input.[Feature_0] AS [Feature_0], keras_input.[Feature_1] AS [Feature_1], keras_input.[Feature_2] AS [Feature_2], keras_input.[Feature_3] AS [Feature_3], keras_input.[Feature_4] AS [Feature_4], keras_input.[Feature_5] AS [Feature_5], keras_input.[Feature_6] AS [Feature_6], keras_input.[Feature_7] AS [Feature_7], keras_input.[Feature_8] AS [Feature_8], keras_input.[Feature_9] AS [Feature_9], keras_input.[Feature_10] AS [Feature_10], keras_input.[Feature_11] AS [Feature_11], keras_input.[Feature_12] AS [Feature_12], keras_input.[Feature_13] AS [Feature_13], keras_input.[Feature_14] AS [Feature_14], keras_input.[Feature_15] AS [Feature_15], keras_input.[Feature_16] AS [Feature_16], keras_input.[Feature_17] AS [Feature_17], keras_input.[Feature_18] AS [Feature_18], keras_input.[Feature_19] AS [Feature_19], keras_input.[Feature_20] AS [Feature_20], keras_input.[Feature_21] AS [Feature_21], keras_input.[Feature_22] AS [Feature_22], keras_input.[Feature_23] AS [Feature_23], keras_input.[Feature_24] AS [Feature_24], keras_input.[Feature_25] AS [Feature_25], keras_input.[Feature_26] AS [Feature_26], keras_input.[Feature_27] AS [Feature_27], keras_input.[Feature_28] AS [Feature_28], keras_input.[Feature_29] AS [Feature_29] 
FROM keras_input), 
keras_input_1_rn AS 
(SELECT row_number() OVER (ORDER BY keras_input_1.[KEY] ASC) AS rn, keras_input_1.[KEY] AS [KEY], keras_input_1.[Feature_0] AS [Feature_0], keras_input_1.[Feature_1] AS [Feature_1], keras_input_1.[Feature_2] AS [Feature_2], keras_input_1.[Feature_3] AS [Feature_3], keras_input_1.[Feature_4] AS [Feature_4], keras_input_1.[Feature_5] AS [Feature_5], keras_input_1.[Feature_6] AS [Feature_6], keras_input_1.[Feature_7] AS [Feature_7], keras_input_1.[Feature_8] AS [Feature_8], keras_input_1.[Feature_9] AS [Feature_9], keras_input_1.[Feature_10] AS [Feature_10], keras_input_1.[Feature_11] AS [Feature_11], keras_input_1.[Feature_12] AS [Feature_12], keras_input_1.[Feature_13] AS [Feature_13], keras_input_1.[Feature_14] AS [Feature_14], keras_input_1.[Feature_15] AS [Feature_15], keras_input_1.[Feature_16] AS [Feature_16], keras_input_1.[Feature_17] AS [Feature_17], keras_input_1.[Feature_18] AS [Feature_18], keras_input_1.[Feature_19] AS [Feature_19], keras_input_1.[Feature_20] AS [Feature_20], keras_input_1.[Feature_21] AS [Feature_21], keras_input_1.[Feature_22] AS [Feature_22], keras_input_1.[Feature_23] AS [Feature_23], keras_input_1.[Feature_24] AS [Feature_24], keras_input_1.[Feature_25] AS [Feature_25], keras_input_1.[Feature_26] AS [Feature_26], keras_input_1.[Feature_27] AS [Feature_27], keras_input_1.[Feature_28] AS [Feature_28], keras_input_1.[Feature_29] AS [Feature_29] 
FROM keras_input_1), 
gru_input_kernel_bias AS 
(SELECT keras_input_1_rn.rn AS rn, keras_input_1_rn.[KEY] AS [KEY], keras_input_1_rn.[Feature_0] AS [Feature_0], keras_input_1_rn.[Feature_1] AS [Feature_1], keras_input_1_rn.[Feature_2] AS [Feature_2], keras_input_1_rn.[Feature_3] AS [Feature_3], keras_input_1_rn.[Feature_4] AS [Feature_4], keras_input_1_rn.[Feature_5] AS [Feature_5], keras_input_1_rn.[Feature_6] AS [Feature_6], keras_input_1_rn.[Feature_7] AS [Feature_7], keras_input_1_rn.[Feature_8] AS [Feature_8], keras_input_1_rn.[Feature_9] AS [Feature_9], keras_input_1_rn.[Feature_10] AS [Feature_10], keras_input_1_rn.[Feature_11] AS [Feature_11], keras_input_1_rn.[Feature_12] AS [Feature_12], keras_input_1_rn.[Feature_13] AS [Feature_13], keras_input_1_rn.[Feature_14] AS [Feature_14], keras_input_1_rn.[Feature_15] AS [Feature_15], keras_input_1_rn.[Feature_16] AS [Feature_16], keras_input_1_rn.[Feature_17] AS [Feature_17], keras_input_1_rn.[Feature_18] AS [Feature_18], keras_input_1_rn.[Feature_19] AS [Feature_19], keras_input_1_rn.[Feature_20] AS [Feature_20], keras_input_1_rn.[Feature_21] AS [Feature_21], keras_input_1_rn.[Feature_22] AS [Feature_22], keras_input_1_rn.[Feature_23] AS [Feature_23], keras_input_1_rn.[Feature_24] AS [Feature_24], keras_input_1_rn.[Feature_25] AS [Feature_25], keras_input_1_rn.[Feature_26] AS [Feature_26], keras_input_1_rn.[Feature_27] AS [Feature_27], keras_input_1_rn.[Feature_28] AS [Feature_28], keras_input_1_rn.[Feature_29] AS [Feature_29], 0.0 + -0.05318124151744763 * keras_input_1_rn.[Feature_0] + 0.20341271169999897 * keras_input_1_rn.[Feature_1] + 0.04271910535274859 * keras_input_1_rn.[Feature_2] + 0.10200743327418205 * keras_input_1_rn.[Feature_3] + -0.03623942197088109 * keras_input_1_rn.[Feature_4] + 0.028543227495632595 * keras_input_1_rn.[Feature_5] + -0.008569731911369172 * keras_input_1_rn.[Feature_6] + 0.3812502852460795 * keras_input_1_rn.[Feature_7] + 0.39873043193010116 * keras_input_1_rn.[Feature_8] + -0.04519150736837263 * keras_input_1_rn.[Feature_9] + -0.2992909618408074 * keras_input_1_rn.[Feature_10] + 0.023795464287658608 * keras_input_1_rn.[Feature_11] + -0.17841728393558967 * keras_input_1_rn.[Feature_12] + -0.19858213839938613 * keras_input_1_rn.[Feature_13] + 0.32295628080079886 * keras_input_1_rn.[Feature_14] + -0.05918887768189024 * keras_input_1_rn.[Feature_15] + 0.11188946528544297 * keras_input_1_rn.[Feature_16] + -0.07492612982602848 * keras_input_1_rn.[Feature_17] + -3.391874004543549e-05 * keras_input_1_rn.[Feature_18] + 0.40706425553852854 * keras_input_1_rn.[Feature_19] + 0.03388503904653345 * keras_input_1_rn.[Feature_20] + -0.3190414480221564 * keras_input_1_rn.[Feature_21] + 0.11474206566595946 * keras_input_1_rn.[Feature_22] + -0.24797890441426598 * keras_input_1_rn.[Feature_23] + 0.13621711963558436 * keras_input_1_rn.[Feature_24] + 0.08303937489569407 * keras_input_1_rn.[Feature_25] + 0.0905756266884285 * keras_input_1_rn.[Feature_26] + -0.16775681742398837 * keras_input_1_rn.[Feature_27] + -0.3649831194086426 * keras_input_1_rn.[Feature_28] + -0.32379157542043 * keras_input_1_rn.[Feature_29] AS dot_prod_z_1, 0.0 + 0.29855956599032196 * keras_input_1_rn.[Feature_0] + -0.054988307263109604 * keras_input_1_rn.[Feature_1] + 0.32383137311338517 * keras_input_1_rn.[Feature_2] + 0.4033587435143404 * keras_input_1_rn.[Feature_3] + 0.1451711136631868 * keras_input_1_rn.[Feature_4] + 0.3715628163170209 * keras_input_1_rn.[Feature_5] + -0.03559080019661354 * keras_input_1_rn.[Feature_6] + -0.08634474158336164 * keras_input_1_rn.[Feature_7] + 0.0552909479719057 * keras_input_1_rn.[Feature_8] + 0.10981421504164246 * keras_input_1_rn.[Feature_9] + 0.3344265490958892 * keras_input_1_rn.[Feature_10] + 0.40563626468605585 * keras_input_1_rn.[Feature_11] + 0.18707849249316122 * keras_input_1_rn.[Feature_12] + 0.3657795052077705 * keras_input_1_rn.[Feature_13] + -0.1651378526104399 * keras_input_1_rn.[Feature_14] + -0.3487734189138024 * keras_input_1_rn.[Feature_15] + 0.05203946144765559 * keras_input_1_rn.[Feature_16] + 0.09728167260010878 * keras_input_1_rn.[Feature_17] + 0.3564021384084294 * keras_input_1_rn.[Feature_18] + -0.1432808004459728 * keras_input_1_rn.[Feature_19] + -0.3461449003816729 * keras_input_1_rn.[Feature_20] + 0.19955280830720523 * keras_input_1_rn.[Feature_21] + 0.369546942044398 * keras_input_1_rn.[Feature_22] + 0.04818127219374041 * keras_input_1_rn.[Feature_23] + -0.35772783507681427 * keras_input_1_rn.[Feature_24] + 0.23742315184121476 * keras_input_1_rn.[Feature_25] + 0.15018057282024033 * keras_input_1_rn.[Feature_26] + 0.11938245297158112 * keras_input_1_rn.[Feature_27] + 0.0787654675546886 * keras_input_1_rn.[Feature_28] + 0.20504387448202488 * keras_input_1_rn.[Feature_29] AS dot_prod_z_2, 0.0 + -0.25255617954056936 * keras_input_1_rn.[Feature_0] + -0.32306939256878026 * keras_input_1_rn.[Feature_1] + -0.3964907164611852 * keras_input_1_rn.[Feature_2] + -0.349034002399779 * keras_input_1_rn.[Feature_3] + -0.2984752793100053 * keras_input_1_rn.[Feature_4] + 0.17933355075416024 * keras_input_1_rn.[Feature_5] + 0.008149470287733218 * keras_input_1_rn.[Feature_6] + -0.3216911031079905 * keras_input_1_rn.[Feature_7] + 0.13971158209522294 * keras_input_1_rn.[Feature_8] + -0.029917695913582565 * keras_input_1_rn.[Feature_9] + 0.2577320245851541 * keras_input_1_rn.[Feature_10] + 0.18446882094511607 * keras_input_1_rn.[Feature_11] + 0.144858288741469 * keras_input_1_rn.[Feature_12] + -0.18064840201843782 * keras_input_1_rn.[Feature_13] + -0.3928520000809304 * keras_input_1_rn.[Feature_14] + -0.21525596448150092 * keras_input_1_rn.[Feature_15] + -0.3611700982535361 * keras_input_1_rn.[Feature_16] + 0.22898276627486536 * keras_input_1_rn.[Feature_17] + -0.047249002336318735 * keras_input_1_rn.[Feature_18] + 0.1676302611036926 * keras_input_1_rn.[Feature_19] + 0.02041648829297893 * keras_input_1_rn.[Feature_20] + 0.37441034193737477 * keras_input_1_rn.[Feature_21] + -0.15293384518496694 * keras_input_1_rn.[Feature_22] + 0.09129691866973488 * keras_input_1_rn.[Feature_23] + 0.17468784066345433 * keras_input_1_rn.[Feature_24] + -0.08594195519507963 * keras_input_1_rn.[Feature_25] + -0.013741563183381211 * keras_input_1_rn.[Feature_26] + -0.34494964913209714 * keras_input_1_rn.[Feature_27] + 0.19512711936681404 * keras_input_1_rn.[Feature_28] + -0.281255940875453 * keras_input_1_rn.[Feature_29] AS dot_prod_r_1, 0.0 + -0.15220703074099318 * keras_input_1_rn.[Feature_0] + 0.06727142545876003 * keras_input_1_rn.[Feature_1] + -0.14708009392282967 * keras_input_1_rn.[Feature_2] + -0.2090345231336693 * keras_input_1_rn.[Feature_3] + 0.40150342870822775 * keras_input_1_rn.[Feature_4] + -0.20559290971439798 * keras_input_1_rn.[Feature_5] + -0.22620633557331282 * keras_input_1_rn.[Feature_6] + 0.09961390333665521 * keras_input_1_rn.[Feature_7] + 0.03465693089255334 * keras_input_1_rn.[Feature_8] + 0.16858431819114605 * keras_input_1_rn.[Feature_9] + 0.22046486410743038 * keras_input_1_rn.[Feature_10] + 0.3208036656483819 * keras_input_1_rn.[Feature_11] + 0.40498491676129467 * keras_input_1_rn.[Feature_12] + 0.13748871808999963 * keras_input_1_rn.[Feature_13] + 0.27714845269684796 * keras_input_1_rn.[Feature_14] + -0.3117629196826158 * keras_input_1_rn.[Feature_15] + -0.06921929349260231 * keras_input_1_rn.[Feature_16] + -0.07922828553375377 * keras_input_1_rn.[Feature_17] + 0.3698809520414288 * keras_input_1_rn.[Feature_18] + -0.03056528879854825 * keras_input_1_rn.[Feature_19] + -0.3786354919785065 * keras_input_1_rn.[Feature_20] + -0.24187492852949186 * keras_input_1_rn.[Feature_21] + -0.06262262124678397 * keras_input_1_rn.[Feature_22] + -0.27945646384103284 * keras_input_1_rn.[Feature_23] + -0.06369063649222279 * keras_input_1_rn.[Feature_24] + -0.3561055952540694 * keras_input_1_rn.[Feature_25] + 0.3150936861310569 * keras_input_1_rn.[Feature_26] + 0.04782750463273627 * keras_input_1_rn.[Feature_27] + 0.12941729049657824 * keras_input_1_rn.[Feature_28] + 0.15450557796078102 * keras_input_1_rn.[Feature_29] AS dot_prod_r_2, 0.0 + -0.2965340518718717 * keras_input_1_rn.[Feature_0] + -0.374292736233577 * keras_input_1_rn.[Feature_1] + -0.1939086595039354 * keras_input_1_rn.[Feature_2] + 0.1313990817668137 * keras_input_1_rn.[Feature_3] + -0.36822245218690536 * keras_input_1_rn.[Feature_4] + -0.07831726553340412 * keras_input_1_rn.[Feature_5] + -0.36521426388704725 * keras_input_1_rn.[Feature_6] + -0.38804846282075695 * keras_input_1_rn.[Feature_7] + -0.2413671457922276 * keras_input_1_rn.[Feature_8] + -0.13906827562605434 * keras_input_1_rn.[Feature_9] + 0.3224460798322434 * keras_input_1_rn.[Feature_10] + -0.28227981882422154 * keras_input_1_rn.[Feature_11] + -0.1727134431524195 * keras_input_1_rn.[Feature_12] + 0.33882018956994964 * keras_input_1_rn.[Feature_13] + -0.11587762030878718 * keras_input_1_rn.[Feature_14] + -0.3512497225121279 * keras_input_1_rn.[Feature_15] + -0.1734441995161414 * keras_input_1_rn.[Feature_16] + -0.02601127367956685 * keras_input_1_rn.[Feature_17] + 0.050149922067205754 * keras_input_1_rn.[Feature_18] + 0.21970917235742382 * keras_input_1_rn.[Feature_19] + 0.1303136431251921 * keras_input_1_rn.[Feature_20] + 0.2621023297072518 * keras_input_1_rn.[Feature_21] + 0.2214833231079829 * keras_input_1_rn.[Feature_22] + -0.296117562223774 * keras_input_1_rn.[Feature_23] + 0.3926640971871155 * keras_input_1_rn.[Feature_24] + 0.05778182241581509 * keras_input_1_rn.[Feature_25] + 0.14811439260989068 * keras_input_1_rn.[Feature_26] + -0.0934562445876656 * keras_input_1_rn.[Feature_27] + 0.19377608365036447 * keras_input_1_rn.[Feature_28] + -0.33914167896220376 * keras_input_1_rn.[Feature_29] AS dot_prod_h_1, 0.0 + 0.17578391976011898 * keras_input_1_rn.[Feature_0] + -0.14531891938513014 * keras_input_1_rn.[Feature_1] + 0.10522884278613265 * keras_input_1_rn.[Feature_2] + -0.1089192931143434 * keras_input_1_rn.[Feature_3] + 0.3002487979493845 * keras_input_1_rn.[Feature_4] + 0.04424304119490602 * keras_input_1_rn.[Feature_5] + 0.172630160817139 * keras_input_1_rn.[Feature_6] + 0.40095373299373593 * keras_input_1_rn.[Feature_7] + -0.029769312416530525 * keras_input_1_rn.[Feature_8] + 0.18795643786163763 * keras_input_1_rn.[Feature_9] + 0.19676128871340381 * keras_input_1_rn.[Feature_10] + -0.3900915663511076 * keras_input_1_rn.[Feature_11] + -0.3837672777680968 * keras_input_1_rn.[Feature_12] + -0.4016762050139887 * keras_input_1_rn.[Feature_13] + 0.16913644631472158 * keras_input_1_rn.[Feature_14] + 0.16509549555087344 * keras_input_1_rn.[Feature_15] + -0.0990075114450919 * keras_input_1_rn.[Feature_16] + 0.24162993205651817 * keras_input_1_rn.[Feature_17] + -0.08281410493413549 * keras_input_1_rn.[Feature_18] + -0.37315892708675474 * keras_input_1_rn.[Feature_19] + 0.015917129453402623 * keras_input_1_rn.[Feature_20] + 0.35700115084503226 * keras_input_1_rn.[Feature_21] + 0.25396850989956654 * keras_input_1_rn.[Feature_22] + 0.08438436761143547 * keras_input_1_rn.[Feature_23] + -0.2535538454636132 * keras_input_1_rn.[Feature_24] + 0.3873411891647244 * keras_input_1_rn.[Feature_25] + 0.006926846995625957 * keras_input_1_rn.[Feature_26] + -0.013630310655449562 * keras_input_1_rn.[Feature_27] + 0.06481363052266431 * keras_input_1_rn.[Feature_28] + 0.24233608614370672 * keras_input_1_rn.[Feature_29] AS dot_prod_h_2 
FROM keras_input_1_rn), 
rnn_cte_gru_1(rn_1, [KEY], [PreviousState_1], [PreviousState_2], [State_1], [State_2]) AS 
(SELECT gru_input_kernel_bias.rn AS rn_1, gru_input_kernel_bias.[KEY] AS [KEY], CAST(0.0 AS FLOAT(53)) AS [PreviousState_1], CAST(0.0 AS FLOAT(53)) AS [PreviousState_2], (1.0 - CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (0.0 + gru_input_kernel_bias.dot_prod_z_1) + 0.5) THEN 1.0 ELSE 0.2 * (0.0 + gru_input_kernel_bias.dot_prod_z_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (0.0 + gru_input_kernel_bias.dot_prod_z_1) + 0.5) THEN 1.0 ELSE 0.2 * (0.0 + gru_input_kernel_bias.dot_prod_z_1) + 0.5 END ELSE 0.0 END) * ((exp(0.0 + gru_input_kernel_bias.dot_prod_h_1) - exp(-(0.0 + gru_input_kernel_bias.dot_prod_h_1))) / (exp(0.0 + gru_input_kernel_bias.dot_prod_h_1) + exp(-(0.0 + gru_input_kernel_bias.dot_prod_h_1)))) AS [State_1], (1.0 - CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (0.0 + gru_input_kernel_bias.dot_prod_z_2) + 0.5) THEN 1.0 ELSE 0.2 * (0.0 + gru_input_kernel_bias.dot_prod_z_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (0.0 + gru_input_kernel_bias.dot_prod_z_2) + 0.5) THEN 1.0 ELSE 0.2 * (0.0 + gru_input_kernel_bias.dot_prod_z_2) + 0.5 END ELSE 0.0 END) * ((exp(0.0 + gru_input_kernel_bias.dot_prod_h_2) - exp(-(0.0 + gru_input_kernel_bias.dot_prod_h_2))) / (exp(0.0 + gru_input_kernel_bias.dot_prod_h_2) + exp(-(0.0 + gru_input_kernel_bias.dot_prod_h_2)))) AS [State_2] 
FROM gru_input_kernel_bias 
WHERE gru_input_kernel_bias.rn = 1 UNION ALL SELECT gru_input_kernel_bias.rn AS rn, gru_input_kernel_bias.[KEY] AS [KEY], CAST(0.0 AS FLOAT(53)) AS [PreviousState_1], CAST(0.0 AS FLOAT(53)) AS [PreviousState_2], CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (0.07889305402131672 * CAST(0.0 AS FLOAT(53)) + -0.12001338771399168 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_1) + 0.5) THEN 1.0 ELSE 0.2 * (0.07889305402131672 * CAST(0.0 AS FLOAT(53)) + -0.12001338771399168 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (0.07889305402131672 * CAST(0.0 AS FLOAT(53)) + -0.12001338771399168 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_1) + 0.5) THEN 1.0 ELSE 0.2 * (0.07889305402131672 * CAST(0.0 AS FLOAT(53)) + -0.12001338771399168 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_1) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + (1.0 - CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (0.07889305402131672 * CAST(0.0 AS FLOAT(53)) + -0.12001338771399168 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_1) + 0.5) THEN 1.0 ELSE 0.2 * (0.07889305402131672 * CAST(0.0 AS FLOAT(53)) + -0.12001338771399168 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (0.07889305402131672 * CAST(0.0 AS FLOAT(53)) + -0.12001338771399168 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_1) + 0.5) THEN 1.0 ELSE 0.2 * (0.07889305402131672 * CAST(0.0 AS FLOAT(53)) + -0.12001338771399168 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_1) + 0.5 END ELSE 0.0 END) * ((exp(0.19308494093258327 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + 0.7665848473643189 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_h_1) - exp(-(0.19308494093258327 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + 0.7665848473643189 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_h_1))) / (exp(0.19308494093258327 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + 0.7665848473643189 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_h_1) + exp(-(0.19308494093258327 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + 0.7665848473643189 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_h_1)))) AS [State_1], CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.034960511057912716 * CAST(0.0 AS FLOAT(53)) + -0.14062053889322335 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.034960511057912716 * CAST(0.0 AS FLOAT(53)) + -0.14062053889322335 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.034960511057912716 * CAST(0.0 AS FLOAT(53)) + -0.14062053889322335 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.034960511057912716 * CAST(0.0 AS FLOAT(53)) + -0.14062053889322335 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_2) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + (1.0 - CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.034960511057912716 * CAST(0.0 AS FLOAT(53)) + -0.14062053889322335 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.034960511057912716 * CAST(0.0 AS FLOAT(53)) + -0.14062053889322335 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.034960511057912716 * CAST(0.0 AS FLOAT(53)) + -0.14062053889322335 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.034960511057912716 * CAST(0.0 AS FLOAT(53)) + -0.14062053889322335 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_z_2) + 0.5 END ELSE 0.0 END) * ((exp(0.8028151871906004 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + -0.4545734405306024 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_h_2) - exp(-(0.8028151871906004 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + -0.4545734405306024 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_h_2))) / (exp(0.8028151871906004 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + -0.4545734405306024 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_h_2) + exp(-(0.8028151871906004 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5) THEN 1.0 ELSE 0.2 * (-0.1888178280225422 * CAST(0.0 AS FLOAT(53)) + -0.24466095906319685 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_1) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + -0.4545734405306024 * CASE WHEN (0.0 <= CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END) THEN CASE WHEN (1.0 <= 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5) THEN 1.0 ELSE 0.2 * (-0.5245068706087566 * CAST(0.0 AS FLOAT(53)) + -0.3341770861035541 * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_r_2) + 0.5 END ELSE 0.0 END * CAST(0.0 AS FLOAT(53)) + gru_input_kernel_bias.dot_prod_h_2)))) AS [State_2] 
FROM gru_input_kernel_bias, rnn_cte_gru_1 
WHERE gru_input_kernel_bias.rn = rnn_cte_gru_1.rn_1 + 1), 
gru_1 AS 
(SELECT rnn_cte_gru_1.[KEY] AS [KEY], CAST(rnn_cte_gru_1.[State_1] AS FLOAT(53)) AS output_1, CAST(rnn_cte_gru_1.[State_2] AS FLOAT(53)) AS output_2 
FROM rnn_cte_gru_1), 
gru_1_1 AS 
(SELECT gru_1.[KEY] AS [KEY], gru_1.output_1 AS output_1, gru_1.output_2 AS output_2 
FROM gru_1), 
layer_dense_1 AS 
(SELECT gru_1_1.[KEY] AS [KEY], -0.4001761601276571 + 0.5362884886393288 * gru_1_1.output_1 + -0.8243281790650232 * gru_1_1.output_2 AS output_1, 0.40017616012765733 + 0.7230070612007538 * gru_1_1.output_1 + -0.3241065120627946 * gru_1_1.output_2 AS output_2 
FROM gru_1_1), 
layer_dense_1_1 AS 
(SELECT layer_dense_1.[KEY] AS [KEY], layer_dense_1.output_1 AS output_1, layer_dense_1.output_2 AS output_2 
FROM layer_dense_1), 
score_soft_max_step1 AS 
(SELECT layer_dense_1_1.[KEY] AS [KEY], layer_dense_1_1.output_1 AS [Score_0], exp(CASE WHEN (100.0 <= CASE WHEN (-100.0 <= layer_dense_1_1.output_1) THEN layer_dense_1_1.output_1 ELSE -100.0 END) THEN 100.0 ELSE CASE WHEN (-100.0 <= layer_dense_1_1.output_1) THEN layer_dense_1_1.output_1 ELSE -100.0 END END) AS [exp_Score_0], layer_dense_1_1.output_2 AS [Score_1], exp(CASE WHEN (100.0 <= CASE WHEN (-100.0 <= layer_dense_1_1.output_2) THEN layer_dense_1_1.output_2 ELSE -100.0 END) THEN 100.0 ELSE CASE WHEN (-100.0 <= layer_dense_1_1.output_2) THEN layer_dense_1_1.output_2 ELSE -100.0 END END) AS [exp_Score_1] 
FROM layer_dense_1_1), 
score_class_union_soft AS 
(SELECT soft_scu.[KEY] AS [KEY], soft_scu.class AS class, soft_scu.[exp_Score] AS [exp_Score] 
FROM (SELECT score_soft_max_step1.[KEY] AS [KEY], 0 AS class, score_soft_max_step1.[exp_Score_0] AS [exp_Score] 
FROM score_soft_max_step1 UNION ALL SELECT score_soft_max_step1.[KEY] AS [KEY], 1 AS class, score_soft_max_step1.[exp_Score_1] AS [exp_Score] 
FROM score_soft_max_step1) AS soft_scu), 
score_soft_max AS 
(SELECT score_soft_max_step1.[KEY] AS [KEY], score_soft_max_step1.[Score_0] AS [Score_0], score_soft_max_step1.[exp_Score_0] AS [exp_Score_0], score_soft_max_step1.[Score_1] AS [Score_1], score_soft_max_step1.[exp_Score_1] AS [exp_Score_1], sum_exp_t.[KEY_sum] AS [KEY_sum], sum_exp_t.[sum_ExpScore] AS [sum_ExpScore] 
FROM score_soft_max_step1 LEFT OUTER JOIN (SELECT score_class_union_soft.[KEY] AS [KEY_sum], sum(score_class_union_soft.[exp_Score]) AS [sum_ExpScore] 
FROM score_class_union_soft GROUP BY score_class_union_soft.[KEY]) AS sum_exp_t ON score_soft_max_step1.[KEY] = sum_exp_t.[KEY_sum]), 
layer_softmax AS 
(SELECT score_soft_max.[KEY] AS [KEY], score_soft_max.[exp_Score_0] / score_soft_max.[sum_ExpScore] AS output_1, score_soft_max.[exp_Score_1] / score_soft_max.[sum_ExpScore] AS output_2 
FROM score_soft_max), 
orig_cte AS 
(SELECT layer_softmax.[KEY] AS [KEY], CAST(NULL AS FLOAT(53)) AS [Score_0], CAST(NULL AS FLOAT(53)) AS [Score_1], layer_softmax.output_1 AS [Proba_0], layer_softmax.output_2 AS [Proba_1], CAST(NULL AS FLOAT(53)) AS [LogProba_0], CAST(NULL AS FLOAT(53)) AS [LogProba_1], CAST(NULL AS BIGINT) AS [Decision], CAST(NULL AS FLOAT(53)) AS [DecisionProba] 
FROM layer_softmax), 
score_class_union AS 
(SELECT scu.[KEY_u] AS [KEY_u], scu.class AS class, scu.[LogProba] AS [LogProba], scu.[Proba] AS [Proba], scu.[Score] AS [Score] 
FROM (SELECT orig_cte.[KEY] AS [KEY_u], 0 AS class, orig_cte.[LogProba_0] AS [LogProba], orig_cte.[Proba_0] AS [Proba], orig_cte.[Score_0] AS [Score] 
FROM orig_cte UNION ALL SELECT orig_cte.[KEY] AS [KEY_u], 1 AS class, orig_cte.[LogProba_1] AS [LogProba], orig_cte.[Proba_1] AS [Proba], orig_cte.[Score_1] AS [Score] 
FROM orig_cte) AS scu), 
score_max AS 
(SELECT orig_cte.[KEY] AS [KEY], orig_cte.[Score_0] AS [Score_0], orig_cte.[Score_1] AS [Score_1], orig_cte.[Proba_0] AS [Proba_0], orig_cte.[Proba_1] AS [Proba_1], orig_cte.[LogProba_0] AS [LogProba_0], orig_cte.[LogProba_1] AS [LogProba_1], orig_cte.[Decision] AS [Decision], orig_cte.[DecisionProba] AS [DecisionProba], max_select.[KEY_m] AS [KEY_m], max_select.[max_Proba] AS [max_Proba] 
FROM orig_cte LEFT OUTER JOIN (SELECT score_class_union.[KEY_u] AS [KEY_m], max(score_class_union.[Proba]) AS [max_Proba] 
FROM score_class_union GROUP BY score_class_union.[KEY_u]) AS max_select ON orig_cte.[KEY] = max_select.[KEY_m]), 
union_with_max AS 
(SELECT score_class_union.[KEY_u] AS [KEY_u], score_class_union.class AS class, score_class_union.[LogProba] AS [LogProba], score_class_union.[Proba] AS [Proba], score_class_union.[Score] AS [Score], score_max.[KEY] AS [KEY], score_max.[Score_0] AS [Score_0], score_max.[Score_1] AS [Score_1], score_max.[Proba_0] AS [Proba_0], score_max.[Proba_1] AS [Proba_1], score_max.[LogProba_0] AS [LogProba_0], score_max.[LogProba_1] AS [LogProba_1], score_max.[Decision] AS [Decision], score_max.[DecisionProba] AS [DecisionProba], score_max.[KEY_m] AS [KEY_m], score_max.[max_Proba] AS [max_Proba] 
FROM score_class_union LEFT OUTER JOIN score_max ON score_class_union.[KEY_u] = score_max.[KEY]), 
arg_max_cte AS 
(SELECT score_max.[KEY] AS [KEY], score_max.[Score_0] AS [Score_0], score_max.[Score_1] AS [Score_1], score_max.[Proba_0] AS [Proba_0], score_max.[Proba_1] AS [Proba_1], score_max.[LogProba_0] AS [LogProba_0], score_max.[LogProba_1] AS [LogProba_1], score_max.[Decision] AS [Decision], score_max.[DecisionProba] AS [DecisionProba], score_max.[KEY_m] AS [KEY_m], score_max.[max_Proba] AS [max_Proba], [arg_max_t_Proba].[KEY_Proba] AS [KEY_Proba], [arg_max_t_Proba].[arg_max_Proba] AS [arg_max_Proba] 
FROM score_max LEFT OUTER JOIN (SELECT union_with_max.[KEY] AS [KEY_Proba], max(union_with_max.class) AS [arg_max_Proba] 
FROM union_with_max 
WHERE union_with_max.[max_Proba] <= union_with_max.[Proba] GROUP BY union_with_max.[KEY]) AS [arg_max_t_Proba] ON score_max.[KEY] = [arg_max_t_Proba].[KEY_Proba])
 SELECT arg_max_cte.[KEY] AS [KEY], arg_max_cte.[Score_0] AS [Score_0], arg_max_cte.[Score_1] AS [Score_1], arg_max_cte.[Proba_0] AS [Proba_0], arg_max_cte.[Proba_1] AS [Proba_1], CASE WHEN (arg_max_cte.[Proba_0] IS NULL OR arg_max_cte.[Proba_0] > 0.0) THEN log(arg_max_cte.[Proba_0]) ELSE -1.79769313486231e+308 END AS [LogProba_0], CASE WHEN (arg_max_cte.[Proba_1] IS NULL OR arg_max_cte.[Proba_1] > 0.0) THEN log(arg_max_cte.[Proba_1]) ELSE -1.79769313486231e+308 END AS [LogProba_1], arg_max_cte.[arg_max_Proba] AS [Decision], arg_max_cte.[max_Proba] AS [DecisionProba] 
FROM arg_max_cte