-- This SQL code was generated by sklearn2sql (development version).
-- Copyright 2018

-- Model : KerasRegressor_Dense
-- Dataset : RandomReg_100
-- Database : pgsql


-- This SQL code can contain one or more statements, to be executed in the order they appear in this file.



-- Model deployment code

WITH keras_input AS 
(SELECT "ADS"."KEY" AS "KEY", "ADS"."Feature_0" AS "Feature_0", "ADS"."Feature_1" AS "Feature_1", "ADS"."Feature_2" AS "Feature_2", "ADS"."Feature_3" AS "Feature_3", "ADS"."Feature_4" AS "Feature_4", "ADS"."Feature_5" AS "Feature_5", "ADS"."Feature_6" AS "Feature_6", "ADS"."Feature_7" AS "Feature_7", "ADS"."Feature_8" AS "Feature_8", "ADS"."Feature_9" AS "Feature_9", "ADS"."Feature_10" AS "Feature_10", "ADS"."Feature_11" AS "Feature_11", "ADS"."Feature_12" AS "Feature_12", "ADS"."Feature_13" AS "Feature_13", "ADS"."Feature_14" AS "Feature_14", "ADS"."Feature_15" AS "Feature_15", "ADS"."Feature_16" AS "Feature_16", "ADS"."Feature_17" AS "Feature_17", "ADS"."Feature_18" AS "Feature_18", "ADS"."Feature_19" AS "Feature_19", "ADS"."Feature_20" AS "Feature_20", "ADS"."Feature_21" AS "Feature_21", "ADS"."Feature_22" AS "Feature_22", "ADS"."Feature_23" AS "Feature_23", "ADS"."Feature_24" AS "Feature_24", "ADS"."Feature_25" AS "Feature_25", "ADS"."Feature_26" AS "Feature_26", "ADS"."Feature_27" AS "Feature_27", "ADS"."Feature_28" AS "Feature_28", "ADS"."Feature_29" AS "Feature_29", "ADS"."Feature_30" AS "Feature_30", "ADS"."Feature_31" AS "Feature_31", "ADS"."Feature_32" AS "Feature_32", "ADS"."Feature_33" AS "Feature_33", "ADS"."Feature_34" AS "Feature_34", "ADS"."Feature_35" AS "Feature_35", "ADS"."Feature_36" AS "Feature_36", "ADS"."Feature_37" AS "Feature_37", "ADS"."Feature_38" AS "Feature_38", "ADS"."Feature_39" AS "Feature_39", "ADS"."Feature_40" AS "Feature_40", "ADS"."Feature_41" AS "Feature_41", "ADS"."Feature_42" AS "Feature_42", "ADS"."Feature_43" AS "Feature_43", "ADS"."Feature_44" AS "Feature_44", "ADS"."Feature_45" AS "Feature_45", "ADS"."Feature_46" AS "Feature_46", "ADS"."Feature_47" AS "Feature_47", "ADS"."Feature_48" AS "Feature_48", "ADS"."Feature_49" AS "Feature_49", "ADS"."Feature_50" AS "Feature_50", "ADS"."Feature_51" AS "Feature_51", "ADS"."Feature_52" AS "Feature_52", "ADS"."Feature_53" AS "Feature_53", "ADS"."Feature_54" AS "Feature_54", "ADS"."Feature_55" AS "Feature_55", "ADS"."Feature_56" AS "Feature_56", "ADS"."Feature_57" AS "Feature_57", "ADS"."Feature_58" AS "Feature_58", "ADS"."Feature_59" AS "Feature_59", "ADS"."Feature_60" AS "Feature_60", "ADS"."Feature_61" AS "Feature_61", "ADS"."Feature_62" AS "Feature_62", "ADS"."Feature_63" AS "Feature_63", "ADS"."Feature_64" AS "Feature_64", "ADS"."Feature_65" AS "Feature_65", "ADS"."Feature_66" AS "Feature_66", "ADS"."Feature_67" AS "Feature_67", "ADS"."Feature_68" AS "Feature_68", "ADS"."Feature_69" AS "Feature_69", "ADS"."Feature_70" AS "Feature_70", "ADS"."Feature_71" AS "Feature_71", "ADS"."Feature_72" AS "Feature_72", "ADS"."Feature_73" AS "Feature_73", "ADS"."Feature_74" AS "Feature_74", "ADS"."Feature_75" AS "Feature_75", "ADS"."Feature_76" AS "Feature_76", "ADS"."Feature_77" AS "Feature_77", "ADS"."Feature_78" AS "Feature_78", "ADS"."Feature_79" AS "Feature_79", "ADS"."Feature_80" AS "Feature_80", "ADS"."Feature_81" AS "Feature_81", "ADS"."Feature_82" AS "Feature_82", "ADS"."Feature_83" AS "Feature_83", "ADS"."Feature_84" AS "Feature_84", "ADS"."Feature_85" AS "Feature_85", "ADS"."Feature_86" AS "Feature_86", "ADS"."Feature_87" AS "Feature_87", "ADS"."Feature_88" AS "Feature_88", "ADS"."Feature_89" AS "Feature_89", "ADS"."Feature_90" AS "Feature_90", "ADS"."Feature_91" AS "Feature_91", "ADS"."Feature_92" AS "Feature_92", "ADS"."Feature_93" AS "Feature_93", "ADS"."Feature_94" AS "Feature_94", "ADS"."Feature_95" AS "Feature_95", "ADS"."Feature_96" AS "Feature_96", "ADS"."Feature_97" AS "Feature_97", "ADS"."Feature_98" AS "Feature_98", "ADS"."Feature_99" AS "Feature_99" 
FROM "RandomReg_100" AS "ADS"), 
keras_input_1 AS 
(SELECT keras_input."KEY" AS "KEY", keras_input."Feature_0" AS "Feature_0", keras_input."Feature_1" AS "Feature_1", keras_input."Feature_2" AS "Feature_2", keras_input."Feature_3" AS "Feature_3", keras_input."Feature_4" AS "Feature_4", keras_input."Feature_5" AS "Feature_5", keras_input."Feature_6" AS "Feature_6", keras_input."Feature_7" AS "Feature_7", keras_input."Feature_8" AS "Feature_8", keras_input."Feature_9" AS "Feature_9", keras_input."Feature_10" AS "Feature_10", keras_input."Feature_11" AS "Feature_11", keras_input."Feature_12" AS "Feature_12", keras_input."Feature_13" AS "Feature_13", keras_input."Feature_14" AS "Feature_14", keras_input."Feature_15" AS "Feature_15", keras_input."Feature_16" AS "Feature_16", keras_input."Feature_17" AS "Feature_17", keras_input."Feature_18" AS "Feature_18", keras_input."Feature_19" AS "Feature_19", keras_input."Feature_20" AS "Feature_20", keras_input."Feature_21" AS "Feature_21", keras_input."Feature_22" AS "Feature_22", keras_input."Feature_23" AS "Feature_23", keras_input."Feature_24" AS "Feature_24", keras_input."Feature_25" AS "Feature_25", keras_input."Feature_26" AS "Feature_26", keras_input."Feature_27" AS "Feature_27", keras_input."Feature_28" AS "Feature_28", keras_input."Feature_29" AS "Feature_29", keras_input."Feature_30" AS "Feature_30", keras_input."Feature_31" AS "Feature_31", keras_input."Feature_32" AS "Feature_32", keras_input."Feature_33" AS "Feature_33", keras_input."Feature_34" AS "Feature_34", keras_input."Feature_35" AS "Feature_35", keras_input."Feature_36" AS "Feature_36", keras_input."Feature_37" AS "Feature_37", keras_input."Feature_38" AS "Feature_38", keras_input."Feature_39" AS "Feature_39", keras_input."Feature_40" AS "Feature_40", keras_input."Feature_41" AS "Feature_41", keras_input."Feature_42" AS "Feature_42", keras_input."Feature_43" AS "Feature_43", keras_input."Feature_44" AS "Feature_44", keras_input."Feature_45" AS "Feature_45", keras_input."Feature_46" AS "Feature_46", keras_input."Feature_47" AS "Feature_47", keras_input."Feature_48" AS "Feature_48", keras_input."Feature_49" AS "Feature_49", keras_input."Feature_50" AS "Feature_50", keras_input."Feature_51" AS "Feature_51", keras_input."Feature_52" AS "Feature_52", keras_input."Feature_53" AS "Feature_53", keras_input."Feature_54" AS "Feature_54", keras_input."Feature_55" AS "Feature_55", keras_input."Feature_56" AS "Feature_56", keras_input."Feature_57" AS "Feature_57", keras_input."Feature_58" AS "Feature_58", keras_input."Feature_59" AS "Feature_59", keras_input."Feature_60" AS "Feature_60", keras_input."Feature_61" AS "Feature_61", keras_input."Feature_62" AS "Feature_62", keras_input."Feature_63" AS "Feature_63", keras_input."Feature_64" AS "Feature_64", keras_input."Feature_65" AS "Feature_65", keras_input."Feature_66" AS "Feature_66", keras_input."Feature_67" AS "Feature_67", keras_input."Feature_68" AS "Feature_68", keras_input."Feature_69" AS "Feature_69", keras_input."Feature_70" AS "Feature_70", keras_input."Feature_71" AS "Feature_71", keras_input."Feature_72" AS "Feature_72", keras_input."Feature_73" AS "Feature_73", keras_input."Feature_74" AS "Feature_74", keras_input."Feature_75" AS "Feature_75", keras_input."Feature_76" AS "Feature_76", keras_input."Feature_77" AS "Feature_77", keras_input."Feature_78" AS "Feature_78", keras_input."Feature_79" AS "Feature_79", keras_input."Feature_80" AS "Feature_80", keras_input."Feature_81" AS "Feature_81", keras_input."Feature_82" AS "Feature_82", keras_input."Feature_83" AS "Feature_83", keras_input."Feature_84" AS "Feature_84", keras_input."Feature_85" AS "Feature_85", keras_input."Feature_86" AS "Feature_86", keras_input."Feature_87" AS "Feature_87", keras_input."Feature_88" AS "Feature_88", keras_input."Feature_89" AS "Feature_89", keras_input."Feature_90" AS "Feature_90", keras_input."Feature_91" AS "Feature_91", keras_input."Feature_92" AS "Feature_92", keras_input."Feature_93" AS "Feature_93", keras_input."Feature_94" AS "Feature_94", keras_input."Feature_95" AS "Feature_95", keras_input."Feature_96" AS "Feature_96", keras_input."Feature_97" AS "Feature_97", keras_input."Feature_98" AS "Feature_98", keras_input."Feature_99" AS "Feature_99" 
FROM keras_input), 
layer_dense_1 AS 
(SELECT keras_input_1."KEY" AS "KEY", -0.07219067960977554 + -0.13183759152889252 * keras_input_1."Feature_0" + -0.4494248926639557 * keras_input_1."Feature_1" + -0.35088178515434265 * keras_input_1."Feature_2" + 0.061673834919929504 * keras_input_1."Feature_3" + 0.27454718947410583 * keras_input_1."Feature_4" + 0.06387992203235626 * keras_input_1."Feature_5" + -0.3787403404712677 * keras_input_1."Feature_6" + -0.2851506769657135 * keras_input_1."Feature_7" + 0.2576981484889984 * keras_input_1."Feature_8" + 0.2798076272010803 * keras_input_1."Feature_9" + -0.004345992114394903 * keras_input_1."Feature_10" + -0.16460616886615753 * keras_input_1."Feature_11" + -0.034379757940769196 * keras_input_1."Feature_12" + 0.1702965497970581 * keras_input_1."Feature_13" + -0.14119356870651245 * keras_input_1."Feature_14" + -0.013850552961230278 * keras_input_1."Feature_15" + 0.2310655564069748 * keras_input_1."Feature_16" + 0.1691669225692749 * keras_input_1."Feature_17" + 0.12683120369911194 * keras_input_1."Feature_18" + -0.2155774086713791 * keras_input_1."Feature_19" + 0.37385687232017517 * keras_input_1."Feature_20" + -0.0724358782172203 * keras_input_1."Feature_21" + 0.03805166855454445 * keras_input_1."Feature_22" + -0.233721524477005 * keras_input_1."Feature_23" + 0.4037245810031891 * keras_input_1."Feature_24" + 0.28922244906425476 * keras_input_1."Feature_25" + 0.16838575899600983 * keras_input_1."Feature_26" + 0.394229918718338 * keras_input_1."Feature_27" + 0.14271031320095062 * keras_input_1."Feature_28" + 0.14706432819366455 * keras_input_1."Feature_29" + 0.08648029714822769 * keras_input_1."Feature_30" + -0.02522437460720539 * keras_input_1."Feature_31" + 0.3559981882572174 * keras_input_1."Feature_32" + 0.30824828147888184 * keras_input_1."Feature_33" + -0.328061580657959 * keras_input_1."Feature_34" + 0.22443713247776031 * keras_input_1."Feature_35" + -0.2743181586265564 * keras_input_1."Feature_36" + -0.42586714029312134 * keras_input_1."Feature_37" + -0.014042247086763382 * keras_input_1."Feature_38" + 0.3330512046813965 * keras_input_1."Feature_39" + -0.05052807182073593 * keras_input_1."Feature_40" + 0.738126277923584 * keras_input_1."Feature_41" + 0.23284904658794403 * keras_input_1."Feature_42" + -0.006468142848461866 * keras_input_1."Feature_43" + -0.22539030015468597 * keras_input_1."Feature_44" + 0.19074636697769165 * keras_input_1."Feature_45" + -0.3371647298336029 * keras_input_1."Feature_46" + 0.1431024670600891 * keras_input_1."Feature_47" + 0.22130660712718964 * keras_input_1."Feature_48" + 0.4496157169342041 * keras_input_1."Feature_49" + -0.043591342866420746 * keras_input_1."Feature_50" + 0.29616373777389526 * keras_input_1."Feature_51" + -0.30524590611457825 * keras_input_1."Feature_52" + -0.07343810796737671 * keras_input_1."Feature_53" + 0.3349483609199524 * keras_input_1."Feature_54" + -0.08626673370599747 * keras_input_1."Feature_55" + -0.040713775902986526 * keras_input_1."Feature_56" + 0.34097468852996826 * keras_input_1."Feature_57" + 0.3156064450740814 * keras_input_1."Feature_58" + -0.15257218480110168 * keras_input_1."Feature_59" + 0.17385241389274597 * keras_input_1."Feature_60" + -0.09542787820100784 * keras_input_1."Feature_61" + -0.27345791459083557 * keras_input_1."Feature_62" + 0.536501944065094 * keras_input_1."Feature_63" + 0.19547055661678314 * keras_input_1."Feature_64" + 0.0542357936501503 * keras_input_1."Feature_65" + 0.02511507086455822 * keras_input_1."Feature_66" + -0.45258238911628723 * keras_input_1."Feature_67" + 0.451455295085907 * keras_input_1."Feature_68" + -0.19576823711395264 * keras_input_1."Feature_69" + 0.2131873518228531 * keras_input_1."Feature_70" + 0.117281474173069 * keras_input_1."Feature_71" + -0.042001873254776 * keras_input_1."Feature_72" + -0.08303039520978928 * keras_input_1."Feature_73" + -0.12371380627155304 * keras_input_1."Feature_74" + 0.21670132875442505 * keras_input_1."Feature_75" + -0.16540659964084625 * keras_input_1."Feature_76" + -0.4507787525653839 * keras_input_1."Feature_77" + 0.23633775115013123 * keras_input_1."Feature_78" + 0.1345949023962021 * keras_input_1."Feature_79" + 0.5129603147506714 * keras_input_1."Feature_80" + 0.23323780298233032 * keras_input_1."Feature_81" + 0.045342326164245605 * keras_input_1."Feature_82" + -0.026840347796678543 * keras_input_1."Feature_83" + 0.07918007671833038 * keras_input_1."Feature_84" + -0.1001354530453682 * keras_input_1."Feature_85" + 0.17013198137283325 * keras_input_1."Feature_86" + 0.13531678915023804 * keras_input_1."Feature_87" + -0.07006832212209702 * keras_input_1."Feature_88" + 0.11601875722408295 * keras_input_1."Feature_89" + 0.14172960817813873 * keras_input_1."Feature_90" + 0.4050861597061157 * keras_input_1."Feature_91" + 0.01458576787263155 * keras_input_1."Feature_92" + -0.07530035823583603 * keras_input_1."Feature_93" + -0.17720474302768707 * keras_input_1."Feature_94" + 0.3461764454841614 * keras_input_1."Feature_95" + -0.0033409767784178257 * keras_input_1."Feature_96" + 0.1409830003976822 * keras_input_1."Feature_97" + -0.16881544888019562 * keras_input_1."Feature_98" + -0.05936839431524277 * keras_input_1."Feature_99" AS output_1, -0.057624801993370056 + -0.30963438749313354 * keras_input_1."Feature_0" + 0.01292562298476696 * keras_input_1."Feature_1" + -0.4082214832305908 * keras_input_1."Feature_2" + 0.1156015694141388 * keras_input_1."Feature_3" + 0.07425744086503983 * keras_input_1."Feature_4" + 0.07563476264476776 * keras_input_1."Feature_5" + -0.16559505462646484 * keras_input_1."Feature_6" + 0.013521065935492516 * keras_input_1."Feature_7" + 0.2589084506034851 * keras_input_1."Feature_8" + 0.1927299052476883 * keras_input_1."Feature_9" + 0.09641360491514206 * keras_input_1."Feature_10" + 0.13894812762737274 * keras_input_1."Feature_11" + 0.3289480209350586 * keras_input_1."Feature_12" + -0.18125107884407043 * keras_input_1."Feature_13" + -0.2590809166431427 * keras_input_1."Feature_14" + 0.32005560398101807 * keras_input_1."Feature_15" + 0.4273287355899811 * keras_input_1."Feature_16" + 0.19616414606571198 * keras_input_1."Feature_17" + 0.028538675978779793 * keras_input_1."Feature_18" + -0.2574779987335205 * keras_input_1."Feature_19" + -0.05753147602081299 * keras_input_1."Feature_20" + 0.25383609533309937 * keras_input_1."Feature_21" + -0.24663998186588287 * keras_input_1."Feature_22" + -0.06779319047927856 * keras_input_1."Feature_23" + 0.1273396611213684 * keras_input_1."Feature_24" + 0.11963145434856415 * keras_input_1."Feature_25" + 0.0888112410902977 * keras_input_1."Feature_26" + 0.1333571821451187 * keras_input_1."Feature_27" + -0.17670071125030518 * keras_input_1."Feature_28" + -0.007168799173086882 * keras_input_1."Feature_29" + -0.12439907342195511 * keras_input_1."Feature_30" + 0.09536711126565933 * keras_input_1."Feature_31" + -0.06123550981283188 * keras_input_1."Feature_32" + 0.34471920132637024 * keras_input_1."Feature_33" + -0.09971616417169571 * keras_input_1."Feature_34" + 0.2223859429359436 * keras_input_1."Feature_35" + -0.17455142736434937 * keras_input_1."Feature_36" + -0.3479938209056854 * keras_input_1."Feature_37" + 0.08803198486566544 * keras_input_1."Feature_38" + 0.36367297172546387 * keras_input_1."Feature_39" + -0.30750390887260437 * keras_input_1."Feature_40" + 0.2457789033651352 * keras_input_1."Feature_41" + -0.0089751360937953 * keras_input_1."Feature_42" + 0.3049672544002533 * keras_input_1."Feature_43" + -0.2243909388780594 * keras_input_1."Feature_44" + -0.17615124583244324 * keras_input_1."Feature_45" + -0.14734570682048798 * keras_input_1."Feature_46" + -0.16723497211933136 * keras_input_1."Feature_47" + 0.07699795067310333 * keras_input_1."Feature_48" + 0.03480931743979454 * keras_input_1."Feature_49" + -0.1828499138355255 * keras_input_1."Feature_50" + -0.07517550885677338 * keras_input_1."Feature_51" + 0.023244708776474 * keras_input_1."Feature_52" + -0.2364947497844696 * keras_input_1."Feature_53" + 0.2749277353286743 * keras_input_1."Feature_54" + -0.2534756362438202 * keras_input_1."Feature_55" + -0.23356249928474426 * keras_input_1."Feature_56" + 0.16848517954349518 * keras_input_1."Feature_57" + 0.28811120986938477 * keras_input_1."Feature_58" + -0.14802078902721405 * keras_input_1."Feature_59" + 0.16326919198036194 * keras_input_1."Feature_60" + -0.07984776794910431 * keras_input_1."Feature_61" + -0.37748095393180847 * keras_input_1."Feature_62" + 0.2769625782966614 * keras_input_1."Feature_63" + 0.17165425419807434 * keras_input_1."Feature_64" + 0.3256412446498871 * keras_input_1."Feature_65" + -0.001013033790513873 * keras_input_1."Feature_66" + -0.41309088468551636 * keras_input_1."Feature_67" + 0.2570115029811859 * keras_input_1."Feature_68" + -0.06655492633581161 * keras_input_1."Feature_69" + -0.024037277325987816 * keras_input_1."Feature_70" + 0.03183535113930702 * keras_input_1."Feature_71" + -0.27023690938949585 * keras_input_1."Feature_72" + 0.051402490586042404 * keras_input_1."Feature_73" + -0.011855082586407661 * keras_input_1."Feature_74" + 0.045479338616132736 * keras_input_1."Feature_75" + -0.14283442497253418 * keras_input_1."Feature_76" + -0.008800572715699673 * keras_input_1."Feature_77" + 0.218351811170578 * keras_input_1."Feature_78" + 0.08807691931724548 * keras_input_1."Feature_79" + 0.3536362051963806 * keras_input_1."Feature_80" + 0.3408975303173065 * keras_input_1."Feature_81" + -0.019709812477231026 * keras_input_1."Feature_82" + -0.04332038760185242 * keras_input_1."Feature_83" + -0.2844012379646301 * keras_input_1."Feature_84" + 0.04581958055496216 * keras_input_1."Feature_85" + -0.2040378600358963 * keras_input_1."Feature_86" + 0.0387020967900753 * keras_input_1."Feature_87" + -0.19170325994491577 * keras_input_1."Feature_88" + 0.11410355567932129 * keras_input_1."Feature_89" + 0.1429116278886795 * keras_input_1."Feature_90" + 0.1728687584400177 * keras_input_1."Feature_91" + 0.20786745846271515 * keras_input_1."Feature_92" + -0.2529102563858032 * keras_input_1."Feature_93" + -0.0007536244229413569 * keras_input_1."Feature_94" + 0.28686827421188354 * keras_input_1."Feature_95" + -0.262301504611969 * keras_input_1."Feature_96" + -0.21624527871608734 * keras_input_1."Feature_97" + 0.11454973369836807 * keras_input_1."Feature_98" + 0.0915466919541359 * keras_input_1."Feature_99" AS output_2, -0.06932196021080017 + -0.27988266944885254 * keras_input_1."Feature_0" + -0.22383198142051697 * keras_input_1."Feature_1" + -0.19779060781002045 * keras_input_1."Feature_2" + 0.03667648136615753 * keras_input_1."Feature_3" + 0.27881094813346863 * keras_input_1."Feature_4" + 0.11670862138271332 * keras_input_1."Feature_5" + -0.23080819845199585 * keras_input_1."Feature_6" + -0.22631309926509857 * keras_input_1."Feature_7" + 0.1971801221370697 * keras_input_1."Feature_8" + 0.1283779740333557 * keras_input_1."Feature_9" + -0.13989326357841492 * keras_input_1."Feature_10" + 0.25571519136428833 * keras_input_1."Feature_11" + 0.0660553127527237 * keras_input_1."Feature_12" + -0.002466923790052533 * keras_input_1."Feature_13" + 0.01837058737874031 * keras_input_1."Feature_14" + 0.0567440502345562 * keras_input_1."Feature_15" + 0.500632643699646 * keras_input_1."Feature_16" + 0.3210309147834778 * keras_input_1."Feature_17" + 0.12770803272724152 * keras_input_1."Feature_18" + -0.34519562125205994 * keras_input_1."Feature_19" + 0.2561115026473999 * keras_input_1."Feature_20" + 0.23506419360637665 * keras_input_1."Feature_21" + -0.2224167436361313 * keras_input_1."Feature_22" + 0.07394822686910629 * keras_input_1."Feature_23" + -0.04969194531440735 * keras_input_1."Feature_24" + 0.30016353726387024 * keras_input_1."Feature_25" + -0.1212259978055954 * keras_input_1."Feature_26" + 0.3897439241409302 * keras_input_1."Feature_27" + 0.13383075594902039 * keras_input_1."Feature_28" + 0.14851315319538116 * keras_input_1."Feature_29" + 0.10913293808698654 * keras_input_1."Feature_30" + 0.367649108171463 * keras_input_1."Feature_31" + 0.3573518693447113 * keras_input_1."Feature_32" + -0.05702560022473335 * keras_input_1."Feature_33" + -0.1668657660484314 * keras_input_1."Feature_34" + -0.08942724764347076 * keras_input_1."Feature_35" + -0.2312365174293518 * keras_input_1."Feature_36" + -0.33582189679145813 * keras_input_1."Feature_37" + 0.012684646993875504 * keras_input_1."Feature_38" + 0.6774035692214966 * keras_input_1."Feature_39" + -0.23886451125144958 * keras_input_1."Feature_40" + 0.4020286798477173 * keras_input_1."Feature_41" + -0.09054567664861679 * keras_input_1."Feature_42" + 0.196971595287323 * keras_input_1."Feature_43" + -0.14817094802856445 * keras_input_1."Feature_44" + 0.0032259714789688587 * keras_input_1."Feature_45" + 0.06300357729196548 * keras_input_1."Feature_46" + 0.1671145111322403 * keras_input_1."Feature_47" + 0.18133370578289032 * keras_input_1."Feature_48" + 0.05149456858634949 * keras_input_1."Feature_49" + -0.14089854061603546 * keras_input_1."Feature_50" + 0.07660000771284103 * keras_input_1."Feature_51" + 0.02551627904176712 * keras_input_1."Feature_52" + -0.13166821002960205 * keras_input_1."Feature_53" + 0.39308565855026245 * keras_input_1."Feature_54" + -0.13322988152503967 * keras_input_1."Feature_55" + -0.07950513064861298 * keras_input_1."Feature_56" + 0.17686912417411804 * keras_input_1."Feature_57" + 0.16160763800144196 * keras_input_1."Feature_58" + -0.07082509249448776 * keras_input_1."Feature_59" + 0.013028948567807674 * keras_input_1."Feature_60" + -0.18651047348976135 * keras_input_1."Feature_61" + -0.28789132833480835 * keras_input_1."Feature_62" + 0.6300216317176819 * keras_input_1."Feature_63" + -0.11763729155063629 * keras_input_1."Feature_64" + 0.21377809345722198 * keras_input_1."Feature_65" + -0.16247643530368805 * keras_input_1."Feature_66" + -0.1579548418521881 * keras_input_1."Feature_67" + 0.36589759588241577 * keras_input_1."Feature_68" + 0.02438327856361866 * keras_input_1."Feature_69" + -0.09791194647550583 * keras_input_1."Feature_70" + 0.024492543190717697 * keras_input_1."Feature_71" + 0.12234864383935928 * keras_input_1."Feature_72" + 0.09006788581609726 * keras_input_1."Feature_73" + -0.11864303797483444 * keras_input_1."Feature_74" + 0.16097481548786163 * keras_input_1."Feature_75" + -0.03432668000459671 * keras_input_1."Feature_76" + -0.18592306971549988 * keras_input_1."Feature_77" + 0.021640073508024216 * keras_input_1."Feature_78" + 0.49134373664855957 * keras_input_1."Feature_79" + 0.5059658288955688 * keras_input_1."Feature_80" + 0.487823486328125 * keras_input_1."Feature_81" + 0.21128717064857483 * keras_input_1."Feature_82" + -0.15691395103931427 * keras_input_1."Feature_83" + -0.2920737862586975 * keras_input_1."Feature_84" + 0.11597368866205215 * keras_input_1."Feature_85" + 0.014009634032845497 * keras_input_1."Feature_86" + 0.09356304258108139 * keras_input_1."Feature_87" + -0.21431244909763336 * keras_input_1."Feature_88" + 0.2298286259174347 * keras_input_1."Feature_89" + 0.16830360889434814 * keras_input_1."Feature_90" + 0.16361305117607117 * keras_input_1."Feature_91" + 0.41765114665031433 * keras_input_1."Feature_92" + -0.40290549397468567 * keras_input_1."Feature_93" + -0.466909259557724 * keras_input_1."Feature_94" + 0.4740399420261383 * keras_input_1."Feature_95" + -0.011191210709512234 * keras_input_1."Feature_96" + -0.0664445012807846 * keras_input_1."Feature_97" + 0.2158496081829071 * keras_input_1."Feature_98" + -0.2296694815158844 * keras_input_1."Feature_99" AS output_3, 0.0550905242562294 + 0.07516402751207352 * keras_input_1."Feature_0" + 0.352203905582428 * keras_input_1."Feature_1" + 0.2390851378440857 * keras_input_1."Feature_2" + 0.027008162811398506 * keras_input_1."Feature_3" + -0.26021480560302734 * keras_input_1."Feature_4" + -0.11229217797517776 * keras_input_1."Feature_5" + 0.3418249785900116 * keras_input_1."Feature_6" + 0.09120139479637146 * keras_input_1."Feature_7" + 0.05920441076159477 * keras_input_1."Feature_8" + -0.23583948612213135 * keras_input_1."Feature_9" + -0.1326601505279541 * keras_input_1."Feature_10" + 0.004927435889840126 * keras_input_1."Feature_11" + -0.2644804120063782 * keras_input_1."Feature_12" + -0.1041976809501648 * keras_input_1."Feature_13" + 0.31411945819854736 * keras_input_1."Feature_14" + 0.046203210949897766 * keras_input_1."Feature_15" + -0.1888575553894043 * keras_input_1."Feature_16" + -0.21822519600391388 * keras_input_1."Feature_17" + -0.12202289700508118 * keras_input_1."Feature_18" + 0.10690397024154663 * keras_input_1."Feature_19" + -0.06437621265649796 * keras_input_1."Feature_20" + 0.08231071382761002 * keras_input_1."Feature_21" + 0.02426844835281372 * keras_input_1."Feature_22" + -0.003659866750240326 * keras_input_1."Feature_23" + -0.07295095920562744 * keras_input_1."Feature_24" + -0.17469552159309387 * keras_input_1."Feature_25" + 0.09452319890260696 * keras_input_1."Feature_26" + -0.08127297461032867 * keras_input_1."Feature_27" + -0.10847882181406021 * keras_input_1."Feature_28" + -0.0010447936365380883 * keras_input_1."Feature_29" + 0.07842861115932465 * keras_input_1."Feature_30" + 0.03260323405265808 * keras_input_1."Feature_31" + -0.18853524327278137 * keras_input_1."Feature_32" + 0.0014027514262124896 * keras_input_1."Feature_33" + -0.1194777637720108 * keras_input_1."Feature_34" + -0.04705791547894478 * keras_input_1."Feature_35" + -0.1428481936454773 * keras_input_1."Feature_36" + 0.22931063175201416 * keras_input_1."Feature_37" + -0.25590139627456665 * keras_input_1."Feature_38" + -0.16673435270786285 * keras_input_1."Feature_39" + 0.23674702644348145 * keras_input_1."Feature_40" + -0.2937285602092743 * keras_input_1."Feature_41" + -0.10230924934148788 * keras_input_1."Feature_42" + 0.08283381909132004 * keras_input_1."Feature_43" + 0.13890941441059113 * keras_input_1."Feature_44" + -0.13244514167308807 * keras_input_1."Feature_45" + -0.09077868610620499 * keras_input_1."Feature_46" + -0.19615167379379272 * keras_input_1."Feature_47" + -0.28247231245040894 * keras_input_1."Feature_48" + -0.015562392771244049 * keras_input_1."Feature_49" + 0.13085594773292542 * keras_input_1."Feature_50" + 0.03994113579392433 * keras_input_1."Feature_51" + -0.08618355542421341 * keras_input_1."Feature_52" + 0.03510451316833496 * keras_input_1."Feature_53" + -0.07430797815322876 * keras_input_1."Feature_54" + 0.07635203003883362 * keras_input_1."Feature_55" + 0.2792645990848541 * keras_input_1."Feature_56" + 0.040735937654972076 * keras_input_1."Feature_57" + -0.16010646522045135 * keras_input_1."Feature_58" + -0.07037516683340073 * keras_input_1."Feature_59" + -0.18565107882022858 * keras_input_1."Feature_60" + -0.10921840369701385 * keras_input_1."Feature_61" + 0.23252762854099274 * keras_input_1."Feature_62" + -0.42493903636932373 * keras_input_1."Feature_63" + -0.00960422120988369 * keras_input_1."Feature_64" + 0.01179138757288456 * keras_input_1."Feature_65" + 0.159966841340065 * keras_input_1."Feature_66" + 0.2943344712257385 * keras_input_1."Feature_67" + -0.1694311797618866 * keras_input_1."Feature_68" + 0.08705198019742966 * keras_input_1."Feature_69" + -0.12097848206758499 * keras_input_1."Feature_70" + 0.05011201649904251 * keras_input_1."Feature_71" + -0.03806852176785469 * keras_input_1."Feature_72" + -0.17432408034801483 * keras_input_1."Feature_73" + -0.0937548279762268 * keras_input_1."Feature_74" + -0.3111637234687805 * keras_input_1."Feature_75" + 0.2136358767747879 * keras_input_1."Feature_76" + -0.006142774596810341 * keras_input_1."Feature_77" + 0.053461529314517975 * keras_input_1."Feature_78" + -0.1605094075202942 * keras_input_1."Feature_79" + -0.3546866774559021 * keras_input_1."Feature_80" + 0.019636841490864754 * keras_input_1."Feature_81" + 0.023116961121559143 * keras_input_1."Feature_82" + -0.19813686609268188 * keras_input_1."Feature_83" + 0.15078240633010864 * keras_input_1."Feature_84" + 0.17162984609603882 * keras_input_1."Feature_85" + -0.11696325242519379 * keras_input_1."Feature_86" + -0.22319482266902924 * keras_input_1."Feature_87" + -0.10548282414674759 * keras_input_1."Feature_88" + -0.2065115123987198 * keras_input_1."Feature_89" + 0.12388886511325836 * keras_input_1."Feature_90" + -0.4033154547214508 * keras_input_1."Feature_91" + -0.32268965244293213 * keras_input_1."Feature_92" + 0.3398466408252716 * keras_input_1."Feature_93" + 0.28031793236732483 * keras_input_1."Feature_94" + -0.6149625182151794 * keras_input_1."Feature_95" + 0.20488817989826202 * keras_input_1."Feature_96" + 0.18678846955299377 * keras_input_1."Feature_97" + 0.11189527809619904 * keras_input_1."Feature_98" + -0.08967601507902145 * keras_input_1."Feature_99" AS output_4 
FROM keras_input_1), 
layer_dense_1_1 AS 
(SELECT layer_dense_1."KEY" AS "KEY", layer_dense_1.output_1 AS output_1, layer_dense_1.output_2 AS output_2, layer_dense_1.output_3 AS output_3, layer_dense_1.output_4 AS output_4 
FROM layer_dense_1), 
layer_dense_2 AS 
(SELECT layer_dense_1_1."KEY" AS "KEY", -0.05088501796126366 + 0.6351051926612854 * layer_dense_1_1.output_1 + 1.223381519317627 * layer_dense_1_1.output_2 + 0.8454029560089111 * layer_dense_1_1.output_3 + -1.2171558141708374 * layer_dense_1_1.output_4 AS output_1 
FROM layer_dense_1_1)
 SELECT layer_dense_2."KEY" AS "KEY", layer_dense_2.output_1 AS "Estimator" 
FROM layer_dense_2