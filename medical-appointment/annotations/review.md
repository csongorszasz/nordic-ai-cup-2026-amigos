# Annotation review

Edit the `decision` column (keep/change-to-absent/change-to-refute).


## Refutes whose quote entails (check bucket) (3)

| question_id | question | quote | note | decision |
|---|---|---|---|---|
| sample_33_hard_no_q03 | Was anything abnormal picked up at the examination? | Nothing abnormal was found on examination. | entailment 0.963 | keep |
| sample_47_hard_no_q06 | Has the throat problem been assessed as a viral infection? | My assessment is tonsillitis. | entailment 0.36 | keep |
| sample_95_hard_no_q03 | Have the results of the blood tests already been gone through? | We have taken your blood test today | entailment 0.454 | keep |

## Refutes that fix an NLI error (confirm near-miss) (10)

| question_id | question | quote | note | decision |
|---|---|---|---|---|
| sample_10_hard_no_q01 | Has the doctor found the condition to be unstable? | I would describe this as a stable situation. | NLI error, entailment 0.0 | keep |
| sample_33_hard_no_q03 | Was anything abnormal picked up at the examination? | Nothing abnormal was found on examination. | NLI error, entailment 0.963 | keep |
| sample_39_hard_no_q01 | Does the pain radiate down the leg? | No, it stays in the lower back. | NLI error, entailment 0.0 | keep |
| sample_43_hard_no_q03 | Is the infection thought to be bacterial? | My assessment is that this is a viral infection. | NLI error, entailment 0.0 | keep |
| sample_47_hard_no_q02 | Were the tonsils pale and free of coatings? | Your tonsils are red and there is coating on them. | NLI error, entailment 0.0 | keep |
| sample_66_hard_no_q02 | Was the patient given a pneumococcal vaccination? | I am here for the influenza vaccination | NLI error, entailment 0.0 | keep |
| sample_71_hard_no_q01 | Did the creatinine turn out to be elevated? | Your creatinine is normal. | NLI error, entailment 0.093 | keep |
| sample_81_hard_no_q05 | Was a bacterial cause considered the most likely one? | most likely a viral gastroenteritis | NLI error, entailment 0.003 | keep |
| sample_92_hard_no_q02 | Has the result of the urine analysis already come back? | We are waiting for that answer. | NLI error, entailment 0.001 | keep |
| sample_95_hard_no_q03 | Have the results of the blood tests already been gone through? | We have taken your blood test today | NLI error, entailment 0.454 | keep |

## Low-confidence hard negatives marked absent (10)

| question_id | question | quote | note | decision |
|---|---|---|---|---|
| sample_47_hard_no_q01 | Does the patient complain of hoarseness as the main symptom? |  | low confidence, absent | keep |
| sample_52_hard_no_q05 | Does the plan involve starting insulin? |  | low confidence, absent | keep |
| sample_55_hard_no_q01 | Does the plan include an urgent referral to a kidney specialist? |  | low confidence, absent | keep |
| sample_56_hard_no_q01 | Is the patient complaining mainly of chest pain? |  | low confidence, absent | keep |
| sample_63_hard_no_q04 | Did the patient describe new episodes of dizziness? |  | low confidence, absent | keep |
| sample_64_hard_no_q04 | Was intense itching the main complaint? |  | low confidence, absent | keep |
| sample_67_hard_no_q02 | Has the patient been referred to a dietitian? |  | low confidence, absent | keep |
| sample_69_hard_no_q02 | Is blood pressure medication being started? |  | low confidence, absent | keep |
| sample_70_hard_no_q03 | Was blood taken during the same visit? |  | low confidence, absent | keep |
| sample_81_hard_no_q04 | Is admission to hospital part of the plan? |  | low confidence, absent | keep |

## Low-confidence off-topic (3)

| question_id | question | quote | note | decision |
|---|---|---|---|---|
| sample_20_off_topic_q01 | At any point does the patient talk about their experiences with a hobby? |  | low confidence, off-topic absent | keep |
| sample_71_off_topic_q02 | According to the conversation, was a specific type of sport played by the patient discussed? |  | low confidence, off-topic absent | keep |
| sample_75_off_topic_q01 | Does the conversation indicate the patient uses specific software? |  | low confidence, off-topic absent | keep |

## Positives the NLI misses (confirm gold span) (10)

| question_id | question | quote | note | decision |
|---|---|---|---|---|
| sample_20_yes_q01 | Was Ibumetin renewed as well? | I am creating prescriptions for both PAMEL and ibumedin now, | NLI missed (p=0.021) | keep |
| sample_23_yes_q06 | Do the changes on the abdomen and lower leg resemble seborrheic keratoses? | They look like seborrheic keratoses. | NLI missed (p=0.032) | keep |
| sample_48_yes_q07 | Does the doctor rule out an endocrine cause? | My assessment is that there is no hormonal cause for the tiredness. | NLI missed (p=0.047) | keep |
| sample_55_yes_q01 | Did the HbA1c come out at 43 mmol/mol? | your long-term sugar value is 43 millimoles per mole is | NLI missed (p=0.013) | keep |
| sample_66_yes_q02 | Is the patient free of any reaction after the injection? | Perfectly fine, thank you. Nothing out of the ordinary. | NLI missed (p=0.027) | keep |
| sample_77_yes_q01 | Does the patient have an infection in the big toe? | The thorn broke the skin, and the infection came after that. An | NLI missed (p=0.002) | keep |
| sample_79_yes_q01 | Did the patient show up in person for the consultation? | Louise, come in. Good to see you. | NLI missed (p=0.268) | keep |
| sample_90_yes_q04 | Does the doctor consider the change harmless? | change looks harmless. | NLI missed (p=0.005) | keep |
| sample_90_yes_q05 | Has the doctor reviewed the photograph? | I have looked at the photograph carefully. | NLI missed (p=0.003) | keep |
| sample_92_yes_q03 | Has a urine sample been sent off for albumin/creatinine ratio analysis? | I'm sending a urine sample for analysis. | NLI missed (p=0.015) | keep |
