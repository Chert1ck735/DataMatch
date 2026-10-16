# Список литературы

Ниже — 66 работ, на которые проект реально опирается: восемь основных, таблица «утверждение → источник»
и полный список использованных работ по темам. Всего при подготовке просмотрено 92 работы; ещё 26 (физический профиль,
WAR и большинство, часть работ по эмпирическому Байесу) дали фон, но в решениях не цитируются. Полный журнал
просмотренной литературы — [`references/references.csv`](references/references.csv).

## Основа метода

| Работа | Что взяли для проекта | Чего источник не доказывает (проверяли сами) |
|---|---|---|
| Routley K., Schulte O. A Markov Game Model for Valuing Player Actions in Ice Hockey. UAI, 2015 | ценность действия через изменение ценности ситуации (L2–L3) | качество нашей калиброванной V |
| Decroos T. et al. Actions Speak Louder than Goals: Valuing Player Actions in Soccer. KDD, 2019 — [doi:10.1145/3292500.3330758](https://doi.org/10.1145/3292500.3330758) | вклад = изменение ожидаемого исхода (L3) | правило деления потери α = 0,5 |
| Efron B., Morris C. Multivariate Empirical Bayes and Estimation of Covariance Matrices. Ann. Statist., 1976 — [doi:10.1214/aos/1176343345](https://doi.org/10.1214/aos/1176343345) | совместная корректировка нескольких показателей (L5) | корректность наших оценённых моментов |
| Morris C. Parametric Empirical Bayes Inference: Theory and Applications. JASA, 1983 — [doi:10.1080/01621459.1983.10477920](https://doi.org/10.1080/01621459.1983.10477920) | эмпирический Байес и интервалы (L5–L6) | покрытие наших интервалов |
| Macdonald B. Adjusted Plus-Minus for NHL Players using Ridge Regression with Goals, Shots, Fenwick, and Corsi. JQAS, 2012 — [doi:10.1515/1559-0410.1447](https://doi.org/10.1515/1559-0410.1447) | RAPM в хоккее (L4) | что RAPM подойдёт при ~4 матчах на игрока |
| Gramacy R., Jensen S., Taddy M. Estimating Player Contribution in Hockey with Regularized Logistic Regression. JQAS, 2013 — [doi:10.1515/jqas-2012-0001](https://doi.org/10.1515/jqas-2012-0001) | большинство игроков измеримо не выделяются (L4) | — |
| Efron B., Tibshirani R. Bootstrap Methods for Standard Errors, Confidence Intervals, and Other Measures of Statistical Accuracy. Statist. Sci., 1986 — [doi:10.1214/ss/1177013815](https://doi.org/10.1214/ss/1177013815) | бутстрэп-интервалы (L6–L7) | полное покрытие при выборе моделей |
| Franks A. et al. Meta-analytics: Tools for Understanding the Statistical Properties of Sports Metrics. JQAS, 2016 — [doi:10.1515/jqas-2016-0098](https://doi.org/10.1515/jqas-2016-0098) | проверка стабильности метрик (L7) | стабильность нашей оценки — её мы проверили сами |

## Какая работа подтверждает какое утверждение

Для каждого утверждения, на которое опирается проект, — работа, из которой оно взято. «Опирается» означает идею
или известный результат; проверку на наших данных мы делали сами (см. [`RESULTS.md`](RESULTS.md)).

### L1 — шанс гола (xG)

| Утверждение | Источник |
|---|---|
| xG в NHL и ridge-APM на xG | [Macdonald (2012)](https://consensus.app/papers/details/ff95e8c0cfff55fa9dd53cb093d9d6b5/) |
| Смещения xG; калибровка по подгруппам | [Davis et al. (2024)](https://doi.org/10.48550/arxiv.2401.09940) |
| Градиентный бустинг для xG NHL (log loss/Brier/AUC) | [Noel (2025)](https://doi.org/10.48550/arxiv.2511.07703) |
| У каждой лиги свой профиль бросков → обучать xG на своих данных | [Csiernik et al. (2025)](https://doi.org/10.3389/fspor.2025.1648099) |
| Дистанция+угол — основа, доп. состояния не дают выигрыша | [Sadeghkhani (2026)](https://doi.org/10.1007/s42081-026-00363-z) |

### L2 — ценность ситуации

| Утверждение | Источник |
|---|---|
| Марковская игра для ценности действий NHL; lookahead важен | [Routley, Schulte (2015)](https://consensus.app/papers/details/4378498102085b7c8e710b12b866f109/) |
| Ценности состояний валидируются по успеху команды | [Schulte et al. (2017)](https://doi.org/10.1007/s10618-017-0496-z) |
| Q-функция, GIM устойчив по сезону | [Liu, Schulte (2018)](https://doi.org/10.24963/ijcai.2018/478) |
| GAM ≈ бустинг; оценивать модель правильной метрикой | [Decroos, Davis (2019)](https://consensus.app/papers/details/a8d32b040a0c5faf8a30bbeeef330769/) |
| Ограничивать контекст, чтобы не утекала личность команды | [Broermann et al. (2026)](https://doi.org/10.48550/arxiv.2608.12926) |

### L3 — вклад действия

| Утверждение | Источник |
|---|---|
| VAEP — вклад = изменение ценности (как наш δ) | [Decroos et al. (2019)](https://doi.org/10.1145/3292500.3330758) |
| Проектные решения меняют рейтинги → фиксировать до расчёта | [Van Roy et al. (2020)](https://consensus.app/papers/details/6652c7628b0e5f6fac1b76a3601d263b/) |

### L4 — поправка на партнёров (RAPM)

| Утверждение | Источник |
|---|---|
| Оценивать по прогнозу будущих матчей | [Sill (2010)](https://consensus.app/papers/details/5d070e0ee22e5d6a8804b155d0e56801/) |
| В хоккее большие ошибки из-за коллинеарности звеньев и редких голов | [Macdonald (2012)](https://doi.org/10.1515/1559-0410.1447) |
| Большинство игроков не выделяются измеримо → поддерживает O2 не принят | [Gramacy et al. (2013)](https://doi.org/10.1515/jqas-2012-0001) |
| RAPM зависит от качества партнёров | [Ghimire et al. (2020)](https://doi.org/10.1371/journal.pone.0237920) |

### L5 — поправка на объём данных

| Утверждение | Источник |
|---|---|
| Многомерный EB, стягивание собственных значений | [Efron, Morris (1976)](https://doi.org/10.1214/aos/1176343345) |
| Основа параметрического EB и интервалов | [Morris (1983)](https://doi.org/10.1080/01621459.1983.10477920) |
| Стягивание по позиции, смеси против пересжатия элиты | [Jensen et al. (2009)](https://doi.org/10.1214/09-ba424) |
| Стягивание собственных значений против вырожденной T_r | [Ledoit, Wolf (2012)](https://doi.org/10.1214/12-aos989) |
| Стягивание по позиции; процессная модель | [Thomas et al. (2013)](https://doi.org/10.1214/13-aoas646) |

### L6 — диапазоны

| Утверждение | Источник |
|---|---|
| Бутстрэп-интервалы | [Efron, Tibshirani (1986)](https://doi.org/10.1214/ss/1177013815) |
| BCa/перцентильные интервалы, покрытие | [DiCiccio, Efron (1996)](https://doi.org/10.1214/ss/1032280214) |
| Выбор метода интервала, проверка симуляцией | [Carpenter, Bithell (2000)](https://doi.org/10.1002/(sici)1097-0258(20000515)19:9<1141::aid-sim479>3.0.co;2-f) |

### L7 — проверка оценки

| Утверждение | Источник |
|---|---|
| Стабильность/различимость/независимость метрик NBA и NHL | [Franks et al. (2016)](https://doi.org/10.1515/jqas-2016-0098) |
| Ограничения ICC/корреляций надёжности | [Warneke et al. (2025)](https://doi.org/10.1007/s00421-025-05720-6) |

### Карточка: роли игроков

| Утверждение | Источник |
|---|---|
| Проверка решения кластеризации по устойчивости | [Lange et al. (2004)](https://doi.org/10.1162/089976604773717621) |
| Выбор числа кластеров бутстрэпом | [Fang, Wang (2012)](https://doi.org/10.1016/j.csda.2011.09.003) |
| Семейство ограниченных ковариаций GMM (mclust), выбор по BIC | [Scrucca et al. (2016)](https://doi.org/10.32614/rj-2016-021) |
| Устойчивость кластеров через бутстрэп | [Yu et al. (2018)](https://doi.org/10.1007/s00180-018-0830-y) |
| Автоматическое выделение ролей игроков | [Behravan et al. (2019)](https://doi.org/10.1089/big.2018.0069) |
| Кластеры игроков внутри амплуа | [Sennaroglu, Tuzkaya (2021)](https://doi.org/10.1177/17543371211062064) |
| Кластеризация игроков по статистике выступлений | [Akhanli, Hennig (2022)](https://doi.org/10.1515/jqas-2022-0037) |
| Экономная структура ковариации GMM для интерпретируемых групп | [Cavicchia et al. (2022)](https://doi.org/10.1007/s11634-021-00488-x) |
| Выбор числа кластеров по устойчивости | [Reizer (2022)](https://doi.org/10.57709/1958395) |
| При неверной форме кластеров GMM даёт худшее разбиение → экономная (общая) ковариация | [Kasa, Rajan (2023)](https://doi.org/10.1038/s41598-023-44608-3) |
| Выбор модели кластеризации | [Rodríguez-Vítores, Matr'an (2023)](https://doi.org/10.1007/s11222-024-10410-y) |
| Свойства кластеризации GMM при анизотропной ковариации | [Chen, Zhang (2024)](https://doi.org/10.52202/079017-3610) |
| Архетипы игроков и состав команды | [Penner (2025)](https://doi.org/10.3389/fspor.2025.1639431) |
| Устойчивые профили игроков при кластеризации | [Pacifico (2025)](https://doi.org/10.1186/s40537-025-01297-1) |
| Тактические архетипы игроков | [Medić et al. (2026)](https://doi.org/10.7906/indecs.24.4.5) |

### Карточка: WAR, удаления, ситуации, вбрасывания, защита

| Утверждение | Источник |
|---|---|
| Пифагорова формула для NHL | [Cochran, Blackstock (2009)](https://doi.org/10.2202/1559-0410.1181) |
| Статистическое обоснование для хоккея | [Dayaratna, Miller (2012)](https://doi.org/10.48550/arxiv.1208.1725) |
| Прямая ценность мала (21 гол на 3509 вбрасываний) | [Liitola (2012)](https://consensus.app/papers/details/ec5931134fc75aaf9bf908c9da57083a/) |
| Показатель = средняя результативность и перевесы → наш G_win по распределению разницы | [Kaplan, Rich (2017)](https://doi.org/10.1515/jqas-2017-0055) |
| Атака опосредует связь владения и защиты | [Riley (2017)](https://doi.org/10.1371/journal.pone.0184346) |
| Процент побед переоценён; ценность — в последствиях и выигрыше «сверх ожидаемого» | [Czuzoj-Shulman et al. (2019)](https://doi.org/10.48550/arxiv.1902.02397) |
| BT в хоккее, байесовская оценка силы | [Whelan, Klein (2020)](https://doi.org/10.13164/ma.2019.09) |
| Индексы по частым событиям (отборы) надёжнее, чем по голам | [Toda et al. (2021)](https://doi.org/10.1371/journal.pone.0263051) |
| Гол в большинстве ≈ гол в равных; PP% 10–30% | [Svarén et al. (2023)](https://doi.org/10.3384/ecp201.5) |
| Bradley–Terry / Elo — стандарт парных сравнений | [Glickman, Stern (2024)](https://doi.org/10.1146/annurev-statistics-040722-061813) |
| 63% голов 5на5, 21% 5на4 | [Luimula (2025)](https://doi.org/10.36950/2025.10ciss006) |

### Шкала и подача результата

| Утверждение | Источник |
|---|---|
| Воспроизводимый WAR с ресэмплингом — аналог нашего подхода | [Yurko et al. (2018)](https://doi.org/10.1515/jqas-2018-0010) |
| Одно число теряет детали — показывать составляющие | [Weber (2018)](https://consensus.app/papers/details/242c9df247705f288f0d4646503a9681/) |
| Распределения помогают неспециалистам | [Kale et al. (2019)](https://doi.org/10.1109/tvcg.2018.2864909) |
| WAR стабилен и понятен клубам | [PFF (2020)](https://consensus.app/papers/details/9703f71473805690a5624deba9ccb2be/) |
| Интервал читают как точное значение — подписывать явно | [Joslyn, Savelli (2021)](https://doi.org/10.3389/fcomp.2020.590232) |
| Quantile dotplot/плотность лучше «усов» и текста | [Castro et al. (2021)](https://doi.org/10.31234/osf.io/wpz8b) |
| Интервал + словесная метка уверенности модели | [Padilla et al. (2021)](https://doi.org/10.3389/fpsyg.2020.579267) |
| Числа и диапазоны понятнее слов; вероятность и уверенность смешивают | [Irwin et al. (2022)](https://doi.org/10.1111/risa.14009) |
| Ширина интервала — сигнал доверия | [Teigen (2022)](https://doi.org/10.1007/s12144-022-03985-0) |
| WAR на 60 минут для игроков с малым временем | [Harrison et al. (2024)](https://doi.org/10.1007/978-3-031-69073-0_5) |
| Совместный показ интервала и метки уверенности | [Fygenson et al. (2026)](https://doi.org/10.48550/arxiv.2608.05487) |

### Применимость в клубе

| Утверждение | Источник |
|---|---|
| Аналитик переводит сложное в понятное | [Callinan et al. (2023)](https://doi.org/10.1371/journal.pone.0280799) |
| Содержательно и просто, не заменяет опыт | [Davidson et al. (2024)](https://doi.org/10.1371/journal.pone.0298346) |
| «сначала решение, потом метрика» | [Kwon (2026)](https://doi.org/10.3389/fspor.2026.1866960) |

## Полный список использованных работ по темам

### Методы по слоям L1–L7 (37)

| Год | Работа | Авторы | Издание | Ссылка | Зачем |
|---|---|---|---|---|---|
| 2026 | Hidden Markov modeling of hockey shot angles with a uniform transient state | Sadeghkhani | Japanese Journal of Statistics and Data Science | [doi:10.1007/s42081-026-00363-z](https://doi.org/10.1007/s42081-026-00363-z) | L1: дистанция+угол — основа, доп. состояния не дают выигрыша |
| 2026 | H-VAEP and H-xT: Valuing Offensive On-the-Ball Actions in Handball | Broermann et al. | arXiv | [doi:10.48550/arxiv.2608.12926](https://doi.org/10.48550/arxiv.2608.12926) | L2/L3: ограничивать контекст, чтобы не утекала личность команды |
| 2025 | Expected by Whom? A Skill-Adjusted Expected Goals Model for NHL Shooters and Goaltenders | Noel | arXiv | [doi:10.48550/arxiv.2511.07703](https://doi.org/10.48550/arxiv.2511.07703) | L1: градиентный бустинг для xG NHL (log loss/Brier/AUC) |
| 2025 | A comparison of shot selection and goal scoring between collegiate and professional women's and men's ice hockey | Csiernik et al. | Frontiers in Sports and Active Living | [doi:10.3389/fspor.2025.1648099](https://doi.org/10.3389/fspor.2025.1648099) | L1: у каждой лиги свой профиль бросков → обучать xG на своих данных |
| 2025 | Discussion on the validity of commonly used reliability indices in sports medicine and exercise science | Warneke et al. | European Journal of Applied Physiology | [doi:10.1007/s00421-025-05720-6](https://doi.org/10.1007/s00421-025-05720-6) | L7: ограничения ICC/корреляций надёжности |
| 2025 | Scoring in the Finnish ice hockey Liiga during the 2023-2024 regular season | Luimula | CISS | [doi:10.36950/2025.10ciss006](https://doi.org/10.36950/2025.10ciss006) | Ситуации: 63% голов 5на5, 21% 5на4 |
| 2024 | Biases in Expected Goals Models Confound Finishing Ability | Davis et al. | arXiv | [doi:10.48550/arxiv.2401.09940](https://doi.org/10.48550/arxiv.2401.09940) | L1: смещения xG; калибровка по подгруппам |
| 2024 | Models and Rating Systems for Head-to-Head Competition | Glickman, Stern | Annual Review of Statistics and Its Application | [doi:10.1146/annurev-statistics-040722-061813](https://doi.org/10.1146/annurev-statistics-040722-061813) | Вбрасывания: Bradley–Terry / Elo — стандарт парных сравнений |
| 2023 | The Importance of Special Teams in Ice Hockey | Svarén et al. | Linköping Hockey Analytics Conf. | [doi:10.3384/ecp201.5](https://doi.org/10.3384/ecp201.5) | Удаления/ситуации: гол в большинстве ≈ гол в равных; PP% 10–30% |
| 2021 | Evaluation of soccer team defense based on prediction models of ball recovery and being attacked | Toda et al. | PLoS ONE | [doi:10.1371/journal.pone.0263051](https://doi.org/10.1371/journal.pone.0263051) | Защита: индексы по частым событиям (отборы) надёжнее, чем по голам |
| 2020 | Valuing On-the-Ball Actions in Soccer: A Critical Comparison of xT and VAEP | Van Roy et al. | — | [ссылка](https://consensus.app/papers/details/6652c7628b0e5f6fac1b76a3601d263b/) | L3: проектные решения меняют рейтинги → фиксировать до расчёта |
| 2020 | Measuring individual worker output in a complementary team setting: Does RAPM isolate individual NBA player contributions? | Ghimire et al. | PLoS ONE | [doi:10.1371/journal.pone.0237920](https://doi.org/10.1371/journal.pone.0237920) | RAPM зависит от качества партнёров |
| 2020 | Prediction and evaluation in College Hockey using the Bradley–Terry–Zermelo model | Whelan, Klein | Mathematics for Application | [doi:10.13164/ma.2019.09](https://doi.org/10.13164/ma.2019.09) | Вбрасывания: BT в хоккее, байесовская оценка силы |
| 2019 | Actions Speak Louder than Goals: Valuing Player Actions in Soccer | Decroos et al. | KDD | [doi:10.1145/3292500.3330758](https://doi.org/10.1145/3292500.3330758) | L3: VAEP — вклад = изменение ценности (как наш δ) |
| 2019 | Interpretable Prediction of Goals in Soccer | Decroos, Davis | — | [ссылка](https://consensus.app/papers/details/a8d32b040a0c5faf8a30bbeeef330769/) | L2: GAM ≈ бустинг; оценивать модель правильной метрикой |
| 2019 | Winning Is Not Everything: A contextual analysis of hockey face-offs | Czuzoj-Shulman et al. | arXiv | [doi:10.48550/arxiv.1902.02397](https://doi.org/10.48550/arxiv.1902.02397) | Вбрасывания: процент побед переоценён; ценность — в последствиях и выигрыше «сверх ожидаемого» |
| 2018 | Deep Reinforcement Learning in Ice Hockey for Context-Aware Player Evaluation | Liu, Schulte | IJCAI | [doi:10.24963/ijcai.2018/478](https://doi.org/10.24963/ijcai.2018/478) | L2/L3: Q-функция, GIM устойчив по сезону |
| 2017 | A Markov Game model for valuing actions, locations, and team performance in ice hockey | Schulte et al. | Data Mining and Knowledge Discovery | [doi:10.1007/s10618-017-0496-z](https://doi.org/10.1007/s10618-017-0496-z) | L2: ценности состояний валидируются по успеху команды |
| 2017 | Decomposing Pythagoras | Kaplan, Rich | JQAS | [doi:10.1515/jqas-2017-0055](https://doi.org/10.1515/jqas-2017-0055) | WAR: показатель = средняя результативность и перевесы → наш G_win по распределению разницы |
| 2017 | Investigating the multivariate nature of NHL player performance with structural equation modeling | Riley | PLoS ONE | [doi:10.1371/journal.pone.0184346](https://doi.org/10.1371/journal.pone.0184346) | Защита: атака опосредует связь владения и защиты |
| 2016 | Meta-analytics: tools for understanding the statistical properties of sports metrics | Franks et al. | JQAS | [doi:10.1515/jqas-2016-0098](https://doi.org/10.1515/jqas-2016-0098) | L7: стабильность/различимость/независимость метрик NBA и NHL |
| 2015 | A Markov Game Model for Valuing Player Actions in Ice Hockey | Routley, Schulte | UAI | [ссылка](https://consensus.app/papers/details/4378498102085b7c8e710b12b866f109/) | L2: марковская игра для ценности действий NHL; lookahead важен |
| 2013 | Estimating player contribution in hockey with regularized logistic regression | Gramacy, Jensen, Taddy | JQAS | [doi:10.1515/jqas-2012-0001](https://doi.org/10.1515/jqas-2012-0001) | RAPM: большинство игроков не выделяются измеримо → поддерживает O2 не принят |
| 2013 | Competing Process Hazard Function Models for Player Ratings in Ice Hockey | Thomas, Ventura, Jensen, Ma | Annals of Applied Statistics | [doi:10.1214/13-aoas646](https://doi.org/10.1214/13-aoas646) | L5/удаления: стягивание по позиции; процессная модель |
| 2012 | An Expected Goals Model for Evaluating NHL Teams and Players | Macdonald | — | [ссылка](https://consensus.app/papers/details/ff95e8c0cfff55fa9dd53cb093d9d6b5/) | L1/L4: xG в NHL и ridge-APM на xG |
| 2012 | Nonlinear Shrinkage Estimation of Large-Dimensional Covariance Matrices | Ledoit, Wolf | Annals of Statistics | [doi:10.1214/12-aos989](https://doi.org/10.1214/12-aos989) | L5/роли: стягивание собственных значений против вырожденной T_r |
| 2012 | Adjusted Plus-Minus for NHL Players using Ridge Regression with Goals, Shots, Fenwick, and Corsi | Macdonald | JQAS | [doi:10.1515/1559-0410.1447](https://doi.org/10.1515/1559-0410.1447) | RAPM: в хоккее большие ошибки из-за коллинеарности звеньев и редких голов |
| 2012 | The Pythagorean Won-Loss Formula and Hockey | Dayaratna, Miller | arXiv | [doi:10.48550/arxiv.1208.1725](https://doi.org/10.48550/arxiv.1208.1725) | WAR: статистическое обоснование для хоккея |
| 2012 | Face-off analysis in men's ice-hockey world championships 2007 | Liitola | — | [ссылка](https://consensus.app/papers/details/ec5931134fc75aaf9bf908c9da57083a/) | Вбрасывания: прямая ценность мала (21 гол на 3509 вбрасываний) |
| 2010 | Improved NBA Adjusted +/- Using Regularization and Out-of-Sample Testing | Sill | MIT Sloan SAC | [ссылка](https://consensus.app/papers/details/5d070e0ee22e5d6a8804b155d0e56801/) | RAPM/валидация: оценивать по прогнозу будущих матчей |
| 2009 | Hierarchical Bayesian Modeling of Hitting Performance in Baseball | Jensen et al. | Bayesian Analysis | [doi:10.1214/09-ba424](https://doi.org/10.1214/09-ba424) | L5: стягивание по позиции, смеси против пересжатия элиты |
| 2009 | Pythagoras and the National Hockey League | Cochran, Blackstock | JQAS | [doi:10.2202/1559-0410.1181](https://doi.org/10.2202/1559-0410.1181) | WAR: пифагорова формула для NHL |
| 2000 | Bootstrap confidence intervals: when, which, what? A practical guide for medical statisticians | Carpenter, Bithell | Statistics in Medicine | [doi:10.1002/(sici)1097-0258(20000515)19:9<1141::aid-sim479>3.0.co;2-f](https://doi.org/10.1002/(sici)1097-0258(20000515)19:9<1141::aid-sim479>3.0.co;2-f) | L6: выбор метода интервала, проверка симуляцией |
| 1996 | Bootstrap confidence intervals | DiCiccio, Efron | Statistical Science | [doi:10.1214/ss/1032280214](https://doi.org/10.1214/ss/1032280214) | L6: BCa/перцентильные интервалы, покрытие |
| 1986 | Bootstrap Methods for Standard Errors, Confidence Intervals, and Other Measures of Statistical Accuracy | Efron, Tibshirani | Statistical Science | [doi:10.1214/ss/1177013815](https://doi.org/10.1214/ss/1177013815) | L6: бутстрэп-интервалы |
| 1983 | Parametric Empirical Bayes Inference: Theory and Applications | Morris | JASA | [doi:10.1080/01621459.1983.10477920](https://doi.org/10.1080/01621459.1983.10477920) | L5: основа параметрического EB и интервалов |
| 1976 | Multivariate Empirical Bayes and Estimation of Covariance Matrices | Efron, Morris | Annals of Statistics | [doi:10.1214/aos/1176343345](https://doi.org/10.1214/aos/1176343345) | L5: многомерный EB, стягивание собственных значений |

### Обзор: интерпретируемость и проверка метрик (14)

| Год | Работа | Авторы | Издание | Ссылка | Зачем |
|---|---|---|---|---|---|
| 2026 | Mixed Uncertainty in One View | Fygenson et al. | arXiv | [doi:10.48550/arxiv.2608.05487](https://doi.org/10.48550/arxiv.2608.05487) | Подача: совместный показ интервала и метки уверенности |
| 2026 | The analytics–practice gap: why sports data fails to translate into coaching decisions | Kwon | Frontiers in Sports and Active Living | [doi:10.3389/fspor.2026.1866960](https://doi.org/10.3389/fspor.2026.1866960) | Применимость: «сначала решение, потом метрика» |
| 2024 | Professional soccer practitioners' perceptions of using performance analysis technology | Davidson et al. | PLOS ONE | [doi:10.1371/journal.pone.0298346](https://doi.org/10.1371/journal.pone.0298346) | Применимость: содержательно и просто, не заменяет опыт |
| 2024 | WAR Accumulation Differential | Harrison et al. | — | [doi:10.1007/978-3-031-69073-0_5](https://doi.org/10.1007/978-3-031-69073-0_5) | Шкала: WAR на 60 минут для игроков с малым временем |
| 2023 | Exploring rugby coaches perception and implementation of performance analytics | Callinan et al. | PLOS ONE | [doi:10.1371/journal.pone.0280799](https://doi.org/10.1371/journal.pone.0280799) | Применимость: аналитик переводит сложное в понятное |
| 2022 | Communicating uncertainty in national security intelligence: verbal and numeric formats | Irwin et al. | Risk Analysis | [doi:10.1111/risa.14009](https://doi.org/10.1111/risa.14009) | Подача: числа и диапазоны понятнее слов; вероятность и уверенность смешивают |
| 2022 | Dimensions of uncertainty communication: verbal terms and numeric ranges | Teigen | Current Psychology | [doi:10.1007/s12144-022-03985-0](https://doi.org/10.1007/s12144-022-03985-0) | Подача: ширина интервала — сигнал доверия |
| 2021 | Visualizing Uncertainty for Non-Expert End Users: The Challenge of the Deterministic Construal Error | Joslyn, Savelli | Frontiers in Computer Science | [doi:10.3389/fcomp.2020.590232](https://doi.org/10.3389/fcomp.2020.590232) | Подача: интервал читают как точное значение — подписывать явно |
| 2021 | Examining Effort in 1D Uncertainty Communication | Castro et al. | IEEE TVCG | [doi:10.31234/osf.io/wpz8b](https://doi.org/10.31234/osf.io/wpz8b) | Подача: quantile dotplot/плотность лучше «усов» и текста |
| 2021 | Uncertain About Uncertainty: qualitative expressions of forecaster confidence | Padilla et al. | Frontiers in Psychology | [doi:10.3389/fpsyg.2020.579267](https://doi.org/10.3389/fpsyg.2020.579267) | Подача: интервал + словесная метка уверенности модели |
| 2020 | PFF WAR: Modeling Player Value in American Football | PFF | — | [ссылка](https://consensus.app/papers/details/9703f71473805690a5624deba9ccb2be/) | Шкала: WAR стабилен и понятен клубам |
| 2019 | Hypothetical Outcome Plots Help Untrained Observers Judge Trends in Ambiguous Data | Kale et al. | IEEE TVCG | [doi:10.1109/tvcg.2018.2864909](https://doi.org/10.1109/tvcg.2018.2864909) | Подача: распределения помогают неспециалистам |
| 2018 | nflWAR: a reproducible method for offensive player evaluation in football | Yurko et al. | JQAS | [doi:10.1515/jqas-2018-0010](https://doi.org/10.1515/jqas-2018-0010) | Шкала: воспроизводимый WAR с ресэмплингом — аналог нашего подхода |
| 2018 | Reconsidering wins above replacement as a metric | Weber | — | [ссылка](https://consensus.app/papers/details/242c9df247705f288f0d4646503a9681/) | Шкала: одно число теряет детали — показывать составляющие |

### Карточка: роли и похожие игроки (15)

| Год | Работа | Авторы | Издание | Ссылка | Зачем |
|---|---|---|---|---|---|
| 2026 | Data-Driven Tactical Archetypes in European Professional Football: A K-Means Clustering Approach | T. Medić и др. | Interdisciplinary Description of Complex Systems | [doi:10.7906/indecs.24.4.5](https://doi.org/10.7906/indecs.24.4.5) | Роли: тактические архетипы игроков |
| 2025 | Player archetypes within basketball: optimizing roster composition to create a championship team | Luke S. J. Penner | Frontiers in Sports and Active Living | [doi:10.3389/fspor.2025.1639431](https://doi.org/10.3389/fspor.2025.1639431) | Роли: архетипы игроков и состав команды |
| 2025 | Fuzzy clustering with robust learning models for soccer player profiling and resilience analysis | Antonio Pacifico | Journal of Big Data | [doi:10.1186/s40537-025-01297-1](https://doi.org/10.1186/s40537-025-01297-1) | Роли: устойчивые профили игроков при кластеризации |
| 2024 | Achieving Optimal Clustering in Gaussian Mixture Models with Anisotropic Covariance Structures | Xin Chen, A. Zhang | Advances in Neural Information Processing Systems 37 | [doi:10.52202/079017-3610](https://doi.org/10.52202/079017-3610) | Роли: свойства кластеризации GMM при анизотропной ковариации |
| 2023 | Avoiding inferior clusterings with misspecified Gaussian mixture models | Siva Rajesh Kasa, Vaibhav Rajan | Scientific Reports | [doi:10.1038/s41598-023-44608-3](https://doi.org/10.1038/s41598-023-44608-3) | Роли: при неверной форме кластеров GMM даёт худшее разбиение → экономная (общая) ковариация |
| 2023 | Improving model choice in classification: an approach based on clustering of covariance matrices | David Rodríguez-Vítores, Carlos Matr'an | Statistics and Computing | [doi:10.1007/s11222-024-10410-y](https://doi.org/10.1007/s11222-024-10410-y) | Роли: выбор модели кластеризации |
| 2022 | Clustering of football players based on performance data and aggregated clustering validity indexes | S. E. Akhanli, C. Hennig | Journal of Quantitative Analysis in Sports | [doi:10.1515/jqas-2022-0037](https://doi.org/10.1515/jqas-2022-0037) | Роли: кластеризация игроков по статистике выступлений |
| 2022 | Gaussian mixture model with an extended ultrametric covariance structure | Carlo Cavicchia и др. | Advances in Data Analysis and Classification | [doi:10.1007/s11634-021-00488-x](https://doi.org/10.1007/s11634-021-00488-x) | Роли: экономная структура ковариации GMM для интерпретируемых групп |
| 2022 | Stability Selection of the Number of Clusters | Gabriella Reizer | — | [doi:10.57709/1958395](https://doi.org/10.57709/1958395) | Роли: выбор числа кластеров по устойчивости |
| 2021 | A cluster analysis of basketball players for each of the five traditionally defined positions | B. Sennaroglu, Gülfem Tuzkaya | Proceedings of the Institution of Mechanical Engineers, Part P: Journal of Sports Engineering and Technology | [doi:10.1177/17543371211062064](https://doi.org/10.1177/17543371211062064) | Роли: кластеры игроков внутри амплуа |
| 2019 | Finding Roles of Players in Football Using Automatic Particle Swarm Optimization-Clustering Algorithm | I. Behravan и др. | Big Data | [doi:10.1089/big.2018.0069](https://doi.org/10.1089/big.2018.0069) | Роли: автоматическое выделение ролей игроков |
| 2018 | Bootstrapping estimates of stability for clusters, observations and model selection | Han Yu и др. | Computational Statistics | [doi:10.1007/s00180-018-0830-y](https://doi.org/10.1007/s00180-018-0830-y) | Роли: устойчивость кластеров через бутстрэп |
| 2016 | mclust 5: Clustering, Classification and Density Estimation Using Gaussian Finite Mixture Models | Luca Scrucca и др. | The R journal | [doi:10.32614/rj-2016-021](https://doi.org/10.32614/rj-2016-021) | Роли: семейство ограниченных ковариаций GMM (mclust), выбор по BIC |
| 2012 | Selection of the number of clusters via the bootstrap method | Yixin Fang, Junhui Wang | Comput. Stat. Data Anal. | [doi:10.1016/j.csda.2011.09.003](https://doi.org/10.1016/j.csda.2011.09.003) | Роли: выбор числа кластеров бутстрэпом |
| 2004 | Stability-Based Validation of Clustering Solutions | Tilman Lange и др. | Neural Computation | [doi:10.1162/089976604773717621](https://doi.org/10.1162/089976604773717621) | Роли: проверка решения кластеризации по устойчивости |
