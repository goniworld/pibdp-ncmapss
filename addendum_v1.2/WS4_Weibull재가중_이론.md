# WS4. Weibull 재가중의 확률적 해석 (초안 v1, 2026-10-28)

GPT의 Q3("함대 사전분포와 Weibull 수명 모델을 같은 엔진에서 추정했는데, 같은 정보를 두 번 쓰는 것 아닌가")에 대한 답이다. 결론부터 적는다.

- **결론 1.** 지금의 재가중(식 9)은 베이지안 갱신으로 해석할 수 있다. 다만 사전분포가 "함대 경로 사전분포 × Weibull"인 **곱 사전분포**로 바뀐 것과 같다.
- **결론 2.** 그 결과 고장 시점 T의 사전 분포가 π₀(T)·f_W(T)가 된다. 두 분포가 같은 개발 엔진들의 수명을 담고 있으므로, **수명 정보를 두 번 센다.** 예측 분포가 실제보다 좁아지는(과신하는) 방향으로 작용한다.
- **결론 3.** 일관된 대안은 Bayesian melding(Poole & Raftery, 2000)의 로그 풀링이다. 가중치를 w ∝ f_W(T)^α / π₀(T)^α 로 바꾸면 된다. α = 1이면 고장 시점의 사전 분포가 정확히 Weibull이 되고, α = 0이면 재가중이 없는 것(noWeibull)이다.
- **결론 4.** 관측된 사실이 결론 2와 맞는다. 보정 전 90% 구간의 시험 포함률이 77.6%로 좁았고, 컨포멀 상수 c₉₀ = 1.842 > 1이 필요했다. 다만 이것만으로 원인을 재가중이라고 단정할 수는 없다. 사전분포 추정 불확실성, 라플라스 근사도 같은 방향으로 작용한다. 그래서 아래 검증 실험이 필요하다.

---

## 1. 설정

- φ = (θ, D): 경로 매개변수 θ = (a, ln b, ln c)와 고장 임계값 D
- 함대 사전분포: p₀(φ) = N(θ; μ_θ, Σ_θ) · N(D; μ_D, σ_D²) (원고 식 6과 3.3절)
- 관측 우도: L(φ) = p(y₁:ₙ | θ). 원고 식 (7)의 첫 항이 −2 ln L에 해당한다.
- 고장 시점은 φ의 결정적 함수이다: T = h(φ) = (τ/c) ln[(D − a)/b] (원고 식 8에서 RUL = T − tₙ).
- π₀(T): p₀(φ)를 h로 밀어낸 고장 시점의 사전 분포(induced prior)
- f_W(T): 개발 엔진의 수명(중도절단 포함)으로 적합한 Weibull 밀도 (원고 식 10)

## 2. 지금 방법이 계산하는 것

PI-BDP는 사후분포 p(φ | y) ∝ L(φ) p₀(φ)에서 표본 φ⁽ⁱ⁾를 뽑고, 가중치 w⁽ⁱ⁾ ∝ f_W(h(φ⁽ⁱ⁾))를 준다. 가중 표본이 나타내는 분포는 다음과 같다.

  p̃(φ | y) ∝ L(φ) · p₀(φ) · f_W(h(φ))

이는 사전분포를 p₀′(φ) ∝ p₀(φ) f_W(h(φ))로 바꾼 **정상적인 베이지안 사후분포**이다. 따라서 "확률적으로 말이 안 되는 결합"은 아니다. 문제는 p₀′가 고장 시점에 대해 무엇을 말하느냐이다. 고장 시점의 사전 분포는 다음과 같다.

  π₀′(T) ∝ π₀(T) · f_W(T)

π₀는 함대 경로 사전분포와 임계값 분포에서 나온다. 둘 다 개발 엔진의 경로와 수명말기 값으로 추정했다. 그래서 π₀ 자체가 이미 개발 엔진의 수명 분포를 대략 담고 있다. f_W도 같은 엔진의 수명에서 추정했다. 두 밀도를 곱하면, 같은 수명 정보가 두 번 들어간 것처럼 분포가 좁아진다. 극단적으로 π₀ ≈ f_W라면 π₀′ ∝ f_W²이 되어, 분산이 대략 절반이 된다.

## 3. 일관된 대안: Bayesian melding의 로그 풀링

Poole & Raftery (2000)는 결정적 모델의 입력에 둔 사전분포가 출력에 유도하는 분포(여기서는 π₀)와, 출력에 대해 따로 가진 사전 정보(여기서는 f_W)를 합치는 방법을 제시했다. 두 분포를 가중치의 합이 1인 로그 풀링으로 합친다.

  q(T) ∝ π₀(T)^(1−α) · f_W(T)^α,  0 ≤ α ≤ 1

그리고 φ의 사전분포를 고장 시점의 분포가 q가 되도록 조정한다. 표본에 주는 가중치로 쓰면 다음과 같다.

  w⁽ⁱ⁾ ∝ [ f_W(T⁽ⁱ⁾) / π₀(T⁽ⁱ⁾) ]^α,  T⁽ⁱ⁾ = h(φ⁽ⁱ⁾)

- α = 0: 재가중 없음 (noWeibull 소거 실험과 같음)
- α = 1: 고장 시점의 사전 분포를 Weibull로 **바꿈**. 경로 사전분포는 "고장 시점이 주어졌을 때의 경로 모양"만 담당한다.
- 지금 방법은 이 틀에서 지수의 합이 2인 경우(π₀¹ · f_W¹)에 해당한다. 즉 가중치의 합이 1이라는 풀링 조건을 어긴다.

π₀(T)는 함대 사전분포에서 φ를 많이 뽑아 T = h(φ)를 계산하고, 커널 밀도로 추정하면 된다. 서브셋마다 한 번만 하면 되는 가벼운 계산이다.

**생존 조건.** 엔진은 tₙ까지 고장 나지 않았다. 엄밀하게는 두 분포 모두 T > tₙ로 조건화해야 한다. 조건화 상수는 모든 표본에 공통이라 가중치 비율에는 영향이 없다. 다만 식 (8)이 T < tₙ인 표본을 RUL = 0으로 자르는 부분은 조건화와 다르다. 원고에 이 근사를 명시한다.

## 4. 검증 실험 (WS2와 함께, 시험 데이터를 쓰지 않음)

개발 데이터의 LOUO-CV(원래 설계 선택에 쓴 것과 같은 폴드)에서 다음 네 가지를 비교한다.

| 변형 | 가중치 |
|---|---|
| noWeibull | 1 |
| 현재 (곱) | f_W(T) |
| melding α = 0.5 | [f_W(T)/π₀(T)]^0.5 |
| melding α = 1 | f_W(T)/π₀(T) |

지표는 다음과 같다.
- RMSE와 수명 구간별 RMSE
- **보정 전** 포함률과 interval score
- 컨포멀 상수 c_α
- CRPS

**예측:** 결론 2가 맞다면 현재 방법이 가장 좁은 구간과 가장 큰 c_α를 보이고, melding 변형은 보정 전 포함률이 명목값에 더 가까울 것이다. 점 예측(RMSE)은 비슷하거나 약간 나빠질 수 있다.

WS2의 합성 데이터 실험에서는 정답 분포를 알기 때문에, 어떤 변형이 참 사후분포에 가까운지 직접 확인할 수 있다.

**원칙.** 이 실험은 V0를 바꾸지 않는다. V0는 사전 등록된 모델로 그대로 두고, 결과는 재가중 단계의 성질을 설명하는 탐색 분석으로 보고한다. 개선된 변형을 새 주 모델로 내세우려면, 시험 데이터를 이미 봤으므로 새로운 자료에서 다시 검증해야 한다.

## 5. 원고에 넣을 문단 초안 (3.4절 끝, 영문)

> **Interpretation of the re-weighting.** Let φ = (θ, D), let T = h(φ) be the failure time implied by Eq. (8), and let π₀(T) be the distribution of T induced by the fleet prior. Weighting posterior samples by f_W(T) targets p(φ | y) f_W(h(φ)), which is the posterior under the modified prior p₀(φ) f_W(h(φ)). The step is therefore a proper Bayesian update, but the prior it implies for the failure time is proportional to π₀(T) f_W(T). Because the fleet prior, the threshold distribution and the Weibull model are all estimated from the same development engines, this product counts the fleet lifetime information twice and tends to make the predictive distribution too narrow. This is consistent with the under-coverage of the uncalibrated intervals (Section 5.3) and is one reason for the conformal step of Section 3.5. A coherent alternative is the logarithmic pooling of Bayesian melding [Poole & Raftery 2000], with weights proportional to [f_W(T)/π₀(T)]^α for 0 ≤ α ≤ 1; Section 5.6.x compares these weightings on the development data.

## 6. 인용 후보 (원고 전에 서지 확인 필요)

- Poole D, Raftery AE. Inference for deterministic simulation models: the Bayesian melding approach. J Am Stat Assoc 2000;95(452):1244–. doi 10.1080/01621459.2000.10474324 — SCIE. 권·호·시작쪽은 저자 공개 PDF로 확인함. 끝쪽과, 3절에 쓴 풀링·가중치 식이 원문 표기와 같은지는 본문 PDF로 확인 필요.
- Genest C, Zidek JV. Combining probability distributions: a critique and an annotated bibliography. Stat Sci 1986;1(1):114–. doi 10.1214/ss/1177013825 — 로그 풀링. 필요하면 인용. 끝쪽 확인 필요.
