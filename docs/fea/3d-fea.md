확인해보니, 실제로 재사용 가치가 있는 3D FEA 결과는 다음과 같습니다.
우선순위	결과	상태	용도
1	[`overnight_force_localized_trend/fea3d`](/home/dk/workspace/lit_ws/output/validation/overnight_force_localized_trend/fea3d)	28개 전부 PASS	가장 완전한 3D 결과 세트
2	[`overnight_24_pair_trend/fea3d`](/home/dk/workspace/lit_ws/output/validation/overnight_24_pair_trend/fea3d)	nominal 1개 PASS	실제 외부 접촉 0.5 mm 검증
3	[`3d_migration/m5_cases`](/home/dk/workspace/lit_ws/output/validation/3d_migration/m5_cases)	6개 PASS	3D contact/migration contract 검증
4	[`calibration_3d`](/home/dk/workspace/lit_ws/output/validation/overnight_force_localized_trend/calibration_3d)	1/2/3/4 N 모두 PASS	힘-변위 및 reaction force calibration
5	[`smoke_3d`](/home/dk/workspace/lit_ws/output/validation/overnight_force_localized_trend/smoke_3d)	12개 PASS	빠른 회귀/시각화 확인용