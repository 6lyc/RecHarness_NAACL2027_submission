SINGLE_MODEL_CONTRACT = '''
MANDATORY SINGLE-MODEL EXPERIMENT: train one independent model per category.
Do not implement ensembles, bagging, boosting of independent predictors, stacking,
multi-seed/checkpoint averaging, score/rank voting, mixtures of independently fitted
experts, or neural scores combined with separately fitted transition/popularity scores.
Each category must load exactly one independently trained model/checkpoint at prediction.
Jointly trained internal layers/features of one network are allowed. Four category
models are necessary for routing and do not constitute an ensemble. All hypotheses,
code generation, debugging, reviews and memory use must respect this constraint even
if an upstream generic prompt suggests an ensemble. The host audits the actual
predictor before accepting validation scores; violations fail the trial and consume
their training budget. Do not read other trials or reuse their trained checkpoints.
'''
