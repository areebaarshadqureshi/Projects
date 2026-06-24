# Error Analysis Report - Urdu Code-Switch Emotion Detector

## Model Strengths
- Overall test macro F1: 0.4305
- Best-performing classes: see class_insights.csv

## Model Weaknesses
- Weakest classes: Fear and Surprise (smallest sample sizes, ~1.1% each)

## Most Confused Class Pairs
| true_class   | predicted_class   |   count |
|:-------------|:------------------|--------:|
| neutral      | anger             |     303 |
| neutral      | happy             |     256 |
| anger        | neutral           |     103 |
| happy        | neutral           |      89 |
| neutral      | sad               |      58 |

## Emoji Dependency Findings
- Not applicable: this dataset contains zero emoji-bearing examples (verified: 0/19,999 rows). The emoji dependency experiment could not be run. This is a documented gap in test coverage.

## Language Dependency Findings
| category   |   mean |   median |    max |   count |
|:-----------|-------:|---------:|-------:|--------:|
| English    | 0.0325 |   0.0158 | 0.9784 |    4571 |
| Other      | 0.0094 |   0      | 0.5572 |    1515 |
| RomanUrdu  | 0.0176 |   0.0112 | 0.851  |     779 |

## Keyword Dependency Findings
- See keyword_sensitivity.csv for negation test results

## Representative Failure Cases
| text                                                                                                                                                                                                                                                                    | true_label   | predicted_label   |   confidence |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------|:------------------|-------------:|
| Esmae gana kaha tha??.. Mujhe to high auto tune or bits kae alaba kuch sunai nahi deya. .Bad song.                                                                                                                                                                      | anger        | sad               |     0.970807 |
| Kay do side hotey positive or negative mein ne apko positve side dikhai negative side ye hai kay unkay sathi b chor hain so disappointing but at least imran khan ko yahn appreciation deni hogi nawaz league ne model town report nai di thi ppp b hoti wo b na krti / | anger        | sad               |     0.967627 |
| Le lo free ab free main maro                                                                                                                                                                                                                                            | anger        | neutral           |     0.945113 |
| blockbuster song yo yo sir Killer video Sabki Phad Ke Rakh Di Aapne love you Yo Yo Honey Singh                                                                                                                                                                          | anger        | happy             |     0.924    |
| Good for you brother... We love you tou aa is sabh sy doir e rahaiye, sy aetabar e uth gya hy, lanat bhe bhejni chaiye jo role play kia inno ny. r py kuch din ghussa rahay ga k in dheaton ko theek q nae kartay                                                       | anger        | happy             |     0.91796  |