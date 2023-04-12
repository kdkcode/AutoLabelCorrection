# AutoLabelCorrection
소프트웨어 종합설계
# 

[소프트웨어 종합설계]

## **자연어 데이터의 Label Correctness 분석 및 대응 방안 연구**

              Offensive Language Detection Dataset 을 대상으로 연구 진행

![Untitled](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/Untitled.png)

팀명 : 오토맞다

조장 : 김도경

조원 : 안현준 이정은

담당 교수 : 한요섭 교수님

담당 조교 : 김영욱

Theory of Computation Lab

---

### 목차

1. **연구주제**

1. **이전 연구와 연구의 필요성**
    
    2.1 현재 데이터의 일반적 문제점
    
    2.2 기존 데이터 레이블링 연구
    
    2.3 연구의 필요성
    

1. **연구내용**
    
    3.1 연구 적용논문 요약
    
    3.2 데이터셋
    
    3.3 실험설정
    
    3.4 연구결과 요약
    

1. **현재 진행상황**

1. **일정 및 역할 배분**

1. **참고논문 및 기타**

---

1. 연구주제

자연어 데이터의 Label Correctness 분석 및 대응 방안 연구를 한다.

이를 위해 Offensive Language Detection Dataset인 Social Bias Frames을 대상으로 연구를 진행하며, Dataset Cartography 논문을 참고하여 training dynamics를 관찰하고, 데이터셋의 샘플을 confidence와 variability의 관점에서 easy-to-learn, ambiguous, hard-to-learn 그룹으로 분류한다. 추가적으로 offensive language detection의 중요한 특성 중 하나인 subjectivity의 관점에서 in-depth analysis를 진행하고, 관련 방법론을 제안하는 것이 목표이다.

1. 연구의 필요성

**2.1 현재 데이터의 일반적 문제점**

![surgeAI 글 인용](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_5.28.47.png)

surgeAI 글 인용

   GoEmotions 데이터셋에 대한 분석 결과를 제공합니다. 이 데이터셋은 Reddit에서 수집된 58,000개의 코멘트를 감성 카테고리에 따라 레이블링한 것입니다. 1,000개의 랜덤한 코멘트를 선택하여, 그 중 308개의 오류를 발견하였습니다. 이 오류는 25개의 코멘트가 레이블링이 잘못되어 있어서 발생한 것으로, negative, neutral, positive 중 하나로 잘못 레이블링된 것입니다. 이러한 오류는 영어 관용어의 이해 부족, 기본 영어 이해 부족, 미국 문화와 정치에 대한 이해 부족, Reddit 미미 등의 이유에서 발생한 것으로 판단됩니다.

이러한 문제들을 해결하기 위해서는 보다 정교한 기술 인프라 연구가 필요하며, 이를 통해 더 나은 데이터셋과 모델이 개발될 수 있을 것입니다.

- 참조링크 : [https://www.surgehq.ai//blog/30-percent-of-googles-reddit-emotions-dataset-is-mislabeled](https://www.surgehq.ai//blog/30-percent-of-googles-reddit-emotions-dataset-is-mislabeled)

**2.2 기존 데이터 레이블링 연구**

 

1) Agreeing to Disagree:  

Annotating Offensive Language Datasets with Annotators’ Disagreement

해당 논문에서는 Offensive Language Detection (OLD)의 주요 문제 중 하나인 주관성(subjectivity)을 다루고 있습니다.

OLD 모델을 학습시키기 위해서는 주관성을 가진 라벨링된 데이터가 필요합니다. 그러나 OLD는 문장의 컨텍스트에 따라 강한 주관성을 가질 수 있습니다. 

따라서 OLD의 라벨링 과정에서 주관성이 포함될 수 있습니다. 이 논문에서는 이러한 주관성을 다루기 위해 annotator 간의 의견이 분열되는 단어와 문장을 분석하고, 이러한 단어와 문장을 포함하는 데이터를 특별히 표시하여 모델 학습에 사용하는 방법을 제안하고 있습니다.

추가적으로 Offensive Language Detection에서 Subjectivity는 어떤 언어가 공격적인지를 판단하는 데 있어서 주관적인 판단이 개입될 수 있는 정도를 말합니다. 즉, 어떤 언어가 공격적이라고 판단되는지에 대한 기준은 사람마다 다를 수 있습니다.

예를 들어, "You're really handsome” 와 "You're really ugly.” 라는 문장이 있다고 가정해보겠습니다. 이 문장이 공격적인지 여부를 판단할 때, 이것이 주관적인 판단일 수 있다는 것을 알 수 있습니다. 어떤 사람은 전자의 문장이 칭찬으로 받아들여지기 때문에 공격적이 아니라고 판단할 수 있지만, 다른 사람은 후자의 문장이 비난으로 받아들여지기 때문에 공격적이라고 판단할 수 있습니다. 이처럼, 언어의 공격성 판단은 객관적인 기준이 아니라 개인의 주관적인 판단이 개입될 수 있기 때문에 Subjectivity가 문제가 될 수 있습니다.

**2.3 연구의 필요성**

 현재 자연어 처리 분야에서는 supervised learning을 기반으로 하는 offensive language detection 방법이 많이 연구되고 있으나, 이를 위해서는 annotated data가 필요하다. 그러나 이러한 데이터의 품질이 좋지 않으면 모델의 성능 및 일반화 능력에 영향을 미칠 수 있다. 따라서 이러한 데이터의 label correctness 분석이 필요하며, 이를 통해 데이터셋의 품질을 높이고 모델의 성능을 향상시키는 방안을 모색하는 것이 필요하다.

1. 연구내용

**3.1 연구적용 논문 요약**

 1) Agreeing to Disagree: 

Annotating Offensive Language Datasets with Annotators’ Disagreement

![스크린샷 2023-04-11 오후 7.27.26.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_7.27.26.png)

본 논문에서는 현재 유해한 언어를 탐지하는 최첨단 접근 방식이 지속적으로 발전하고 있는 소셜 미디어 시나리오에 빠르게 적응할 수 있는 것이 중요하다는 것을 강조하며, 이를 위해 알고리즘적인 측면에서 문제를 해결하는 방법이 제안되어왔지만, 이러한 데이터의 질적인 측면에 대한 연구는 덜 이루어졌다는 것을 지적합니다. 본 연구는 최근 떠오르고 있는 추세에 따라, 주관성이 높은 offensive 언어 데이터셋을 만드는 동안 annotators 간 동의 수준에 초점을 맞추며 데이터를 선택합니다. 영어 트윗의 세 가지 주제를 다루는 새로운 데이터셋을 만들고, 각각 다섯 가지의 crowd-sourced 판단을 수집합니다. 

![스크린샷 2023-04-11 오후 7.27.59.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_7.27.59.png)

또한, annotators의 동의 수준에 따라 훈련 및 테스트 데이터를 선택하는 것이 분류기의 성능과 강건성에 강력한 영향을 미치는 것을 보여주는 많은 실험을 제시합니다. 이러한 결과는 교차 도메인 실험에서도 검증되며, 인기 있는 벤치마크 데이터셋을 사용하여 연구합니다. annotators의 동의가 낮은 hard cases가 반드시 annotation의 질이 나쁜 것이 아니며, 미래 데이터셋에서는 더 많은 ambiguous cases가 포함되어야 하며, 특히 테스트 세트에서 다양한 의견을 반영할 수 있도록 해야 한다는 주장을 합니다.

2) [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://aclanthology.org/2020.emnlp-main.746.pdf)

이 논문은 대규모 데이터셋에서 데이터의 질을 평가하는 것이 어렵다는 문제점을 해결하기 위해 데이터 맵(Data Maps)이라는 모델 기반 도구를 제안합니다. 이를 위해 모델의 학습 동안 각각의 인스턴스에 대한 동작(학습 동력)을 활용하여 데이터 맵을 구성합니다. 이것은 각 예제에 대한 모델의 신뢰도와 이 신뢰도의 epoch 간 변동성이라는 두 가지 직관적인 측정치를 단일 훈련 실행에서 얻을 수 있도록 합니다. 4개 데이터셋을 통한 실험 결과, 이러한 모델 종속적인 측정치는 데이터 맵에서 세 가지 차이점 있는 지역을 보여주는 것으로 나타났습니다. 첫째, 모델과 관련하여 모호한 지역은 분포가 다른 일반화에 가장 크게 기여합니다. 둘째, 데이터에서 가장 많은 지역은 모델이 쉽게 학습할 수 있으며, 모델 최적화에서 중요한 역할을 합니다. 마지막으로, 데이터 맵은 모델이 학습하기 어려운 인스턴스가 포함된 지역을 발견합니다. 이러한 인스턴스는 종종 레이블링 오류와 일치합니다. 결과적으로, 데이터의 양에서 질로의 전환은 강력한 모델 및 분포 이외의 일반화 향상을 이끌어 낼 수 있다는 것을 보여줍니다.

![Data map for SNL train set, based on a RoBERTA-large classifier<그림1>](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25EC%25BA%25A1%25EC%25B2%2598.png)

Data map for SNL train set, based on a RoBERTA-large classifier<그림1>

 *그래프 요약 :

 datamap의 confidence와 variability 사이에 음의 상관 관계(negative correlation)가 있다는 결론을 내렸습니다. 즉, confidence가 높을수록 variability가 낮고, confidence가 낮을수록 variability가 높다는 것을 의미합니다. 이는 라벨링 오류와 같은 노이즈 요인이 있는 데이터 포인트일수록 variability가 높아지기 때문입니다. 따라서, datamap에서 가장 variability가 높은 데이터 포인트들을 식별하고, 이를 다시 라벨링하여 데이터셋의 정확성을 향상시키는 것이 중요하다는 결론을 내리고 있습니다. SBIC 데이터셋에서도 같은 경향을 보이는지 확인해보아야 합니다.

*Training Dynamics란?

"Training Dynamics"는 모델을 학습시키는 동안 데이터셋의 특성을 나타내는 용어입니다. 이 용어는 데이터셋의 특징과 모델의 학습 과정 간의 상호작용을 나타내며, 학습 동안 데이터셋이 어떻게 변화하는지를 이해하는 데 도움이 됩니다.

데이터셋의 training dynamic을 측정하면 모델 학습 중 발생하는 문제를 이해하고 수정하는 데 도움이 됩니다. 예를 들어, 학습 동안 데이터셋의 특정 부분이 잘못 레이블링되어 있거나 모델이 이를 이용하여 편향되게 학습되는 경우가 있을 수 있습니다. 이러한 문제를 발견하고 수정함으로써 모델의 성능을 향상시킬 수 있습니다.

데이터셋의 training dynamic을 분석하기 위해, 모델의 학습 과정을 시각화하고, 데이터셋의 다양한 부분을 살펴보며, 모델의 예측 결과와 실제 결과의 차이를 분석합니다. 이를 통해 데이터셋의 특성을 파악하고 모델 학습 동안 발생하는 문제를 이해할 수 있습니다.

**3.2 데이터셋**

 1) SBIC 데이터

SBIC는 Social Bias In Context(SBIC)의 약어로, 미국의 일부 지역에서 모은 대화 데이터를 바탕으로 만들어진 데이터셋 입니다. 이 데이터셋은 대화 내용에서 나타난 특정 키워드(예: "이슬람", "여성", "성적 소수자", "아프리카계 미국인" 등)를 포함하는 문장들을 선별한 후, 해당 문장들이 포함된 대화의 전후 문맥(총 3-4문장)을 함께 제공하는 형태로 구성되어 있습니다.

SBIC 데이터셋은 기계학습 분야에서 인식되는 바와 같이, 모델의 공정성과 다양성을 보장하기 위해 만들어졌습니다. 이를 위해 데이터셋은 미국의 인구통계학적 정보와 대조적인 특성을 가진 다양한 인물들에게서 수집되었으며, 각 대화의 발화자들도 인종, 성별, 종교, 나이, 교육 수준 등 다양한 인적 특성을 가지고 있습니다. 따라서 SBIC 데이터셋은 다양한 사회적, 문화적 배경에서 발화된 대화의 특성을 고려하여 모델의 공정성을 개선하고 다양성을 확보하기 위한 중요한 자원으로 인식됩니다.

![Example of SBIC 데이터셋](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_7.33.57.png)

Example of SBIC 데이터셋

**3.3 실험설정**

3.3.1 

1) model : bert-base-uncased

기존 논문에서는 RoBERTa를 사용하였지만, RoBERTa는 더 많은 학습 데이터와 높은 컴퓨팅 리소스를 필요로 한다. 우리가 연구하는 SBIC 데이터의 특성과 규모를 고려하였을 때  많은 양의 리소스를 필요로 하지 않으므로 BERT를 사용한다.

2) seed : 789

3) epoch : 6(5에서 수렴)

4) learning rate : 1e-5

5) batch size : 96

6) Size of train set : 35424
    Size of validation set : 4666
    Size of test set : 4691

3.3.2

1) model : bert-base-uncased

2) seed : 789

3) epoch : 6(3에서 수렴)

4) learning rate : 5e-6

5) batch size : 8

6) Size of train set : 35424
    Size of validation set : 4666
    Size of test set : 4691

**3.4 연구결과 요약(팀원)**

![<figure1>3.3.1](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25E1%2584%2589%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A9%25E1%2586%25BC_%25E1%2584%258B%25E1%2585%25B5%25E1%2584%2586%25E1%2585%25B5%25E1%2584%258C%25E1%2585%25B51.png)

<figure1>3.3.1

![스크린샷 2023-04-11 오후 8.44.19.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_8.44.19.png)

x축은 variability, y축은 confidence를 나타내고 있으며 각 점의 색과 모양은 correctness를 나타냅니다. variability는 낮으면서 높은 confidence를 보이는 영역을 easy-to-learn, 낮은 variability와 낮은 confidence를 보이는 영역을 hard-to-learn, 높은 variability를 보이는 영역을 ambiguous로 분류했습니다. 옆에는 세 개의 영역(variability, confidence, correctness)에 대해 밀집분포(density plot)가 각각 그려져 있습니다.

<figure 1>은 epoch을 6으로 설정하여 학습시킨 결과입니다.  0.5에서 0.75 사이의 confidence(high confidence)와 0.0과 0.2 사이의 variability(low variability)에 데이터가 밀집되어 있는 것을 볼 수 있습니다. 이 실험에서는 epoch이 5일때 수렴하는 것으로 보이며 이때 정확도는 0.55 정도입니다.

![<figure 2>3.3.2](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25E1%2584%2589%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A9%25E1%2586%25BC_img2.png)

<figure 2>3.3.2

![스크린샷 2023-04-12 오전 3.56.23.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-12_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_3.56.23.png)

<figure 2> 역시 epoch을 6으로 설정한 실험의 결과입니다.  이 실험에서는 epoch이 3에서 수렴하며 이 때의 정확도는 약 0.65입니다. <figure 1>과 비슷하게  high confidence 영역과 low variability 영역에 데이터가 밀집되어 있는 것을 볼 수 있습니다.  하지만 correctness의 밀집 분포(density plot)를 보면 <figure 1>과 다르게 correctness가 0.0부터 1.0까지 고르게 나타나고 있습니다.

![스크린샷 2023-04-11 오후 7.32.16.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%20bffb81af039d448da486fe1d0bfda064/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-04-11_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_7.32.16.png)

<figure3>Data map for SNL train set, based on a RoBERTA-large classifier

1. 현재 진행상황

논문 Agreeing to Disagree와 Dataset Cartography를 참고하여 SBIC 데이터를 전처리하여 offensive와 post를 남기고, train dynamics를 구현해보았다. 이를 통해 offensive language detection dataset에 적용해보았으며, 데이터셋의 샘플을 confidence와 variability의 관점에서 easy-to-learn, ambiguous, hard-to-learn 그룹으로 분류하며, subjectivity의 관점에서 in-depth analysis를 진행하고자 한다. 이를 통해 데이터의 label correctness를 분석하고, 데이터셋의 품질을 높이는 방안을 모색하는 것이 목표이다.

1. 일정 및 역할 배분

[오토맞다 Task Plan](https://www.notion.so/b63d1d0c98204aa0a09eab974244c20e)

- 논문스터디 및 연구주제 구체화
- (-3월말) SBIC 데이터셋에 dataset cartography 적용
- Trainin dynamics datamap을 기반으로 depth analysis 진행
- (~4월중순) 중간보고서 작성
- [ ]  (4월말-5월 초) 소프트웨어 종합설계 중간 발표
- [ ]  (5월중순) subjectivity 측면에서 in-depth analysis 진행
- [ ]  (-5월말) subjectivity 측면에서 in-depth analysis 진행 및 방법론 제안
- [ ]  (6월) 전시회 진행

- 지금까지 진행한 파트
    - 김도경 : Cartography 설계 및 환경변수 테스트, Depth Analysis, Thesis Study
    - 안현준 : Cartography 설계 및 datamap 도출, Depth Analysis
    - 이정은 : Cartography 설계, Depth Analysis

1. 참고논문 및 기타

1. [https://www.surgehq.ai//blog/30-percent-of-googles-reddit-emotions-dataset-is-mislabeled](https://www.surgehq.ai//blog/30-percent-of-googles-reddit-emotions-dataset-is-mislabeled)
2. **Interactive Label Cleaning with Example-based Explanations (NeuRIPS 2021,** [https://proceedings.neurips.cc/paper/2021/file/6c349155b122aa8ad5c877007e05f24f-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/6c349155b122aa8ad5c877007e05f24f-Paper.pdf)**)**
3. **Clean or Annotate:How to Spend a Limited Data Collection Budget (DEEPLO 2022,** [https://aclanthology.org/2022.deeplo-1.17.pdf](https://aclanthology.org/2022.deeplo-1.17.pdf))
4. **Agreeing to Disagree: Annotating Offensive Language Datasets with Annotators’ Disagreement (EMNLP 2021,** [https://aclanthology.org/2021.emnlp-main.822/](https://aclanthology.org/2021.emnlp-main.822/)**)**
5. **Dataset Cartography:Mapping and Diagnosing Datasets with Training Dynamics (EMNLP 2020,** [https://aclanthology.org/2020.emnlp-main.746/](https://aclanthology.org/2020.emnlp-main.746/)**)**
6. **Social Bias Frames: Reasoning about Social and Power Implications of Language (EMNLP 2020,** [https://aclanthology.org/2020.acl-main.486/](https://aclanthology.org/2020.acl-main.486/)**)**
