# PlantVillage Leaf-Disease Classification

PyTorch로 PlantVillage 데이터셋(38클래스, 약 54,000장)의 잎 질병을 분류하는 딥러닝 팀 프로젝트.
Baseline CNN에서 출발해 4가지 개선 기법을 **동일 조건 3회 반복 실행 평균**으로 비교했다.

## 결과 요약

| 모델/기법 | 평균 Test Acc | Baseline 대비 |
|---|---:|---:|
| Baseline CNN | 79.75% ± 1.21 | 기준 |
| + Data Augmentation | 86.99% ± 0.59 | +7.24%p |
| Transfer Learning (Feature Extraction) | 88.55% ± 0.33 | +8.80%p |
| Transfer Learning (Fine-Tuning) | 88.63% ± 0.06 | +8.88%p |
| Residual CNN | 89.68% ± 0.85 | +9.93%p |

핵심 결론: 최고 성능은 Residual CNN, 안정성(표준편차)은 Fine-Tuning이 가장 우수.
자세한 해석과 그래프는 노트북 하단의 실험 결론 섹션 참고.

## 내 역할 (김민우)

- Baseline 정리, **Data Augmentation** 및 **Transfer Learning (Feature Extraction)** 실험 담당
- 실험 통합·결론 작성, 최종 점검 (팀원 김민엽: Residual CNN, Fine-Tuning)

## 실행 방법

1. Google Colab에서 노트북 열기 (런타임 → T4 GPU 선택)
2. 셀을 위에서부터 순서대로 실행 — 데이터셋은 gdown으로 자동 다운로드
3. 각 실험 섹션(AUG / RES / TL-FE / TL-FT)은 독립적으로 실행 가능

## 기술 스택

Python · PyTorch · torchvision · scikit-learn · matplotlib
