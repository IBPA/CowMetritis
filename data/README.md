## Cow Metritis Cure Risk Study Dataset


1. Filename: *MetritisCureRiskStudy.csv*

2. Number of instances: 550

3. Number of attributes: 24 including the dependent variables

4. Attribute information

|                |  categorical  |  non-categorical  |
| -------------- | :-----------: | :---------------: |
| **numeric**    | 8             | 10                |
| **string**     | 6             | 0                 |

```
Dairy: (string, categorical) Name of the farm
ID: (numeric) Identification number of the animal
DIMd0: (numeric) Days in milk at the enrolment. Days when metritis was diagnosed, and the animal was assigned to receive the antibiotic or not. Day in milk start when a cow has a parturition and initiates a lactation. Metritis generally will occur within the first two post parturition.
TRT: (string, categorical) Treatment
	- CEF: Antibiotic group
	- CON: Non treated group
D0month: (numeric, categorical) Month of the metritis diagnosis
Season: (string, categorical) Season of the metritis diagnosis
	- cool: Metritis was diagnosed between September and May
	- hot: Metritis was diagnosed between June and August
Lact: (numeric) Number of birth followed by lactation that the each cow had
Parity: (string, categorical) Number of birth
	- P: Primiparous (one birth)
	- M: Multiparous (more than one birth)
CalvDif: (numeric, categorical) How hard was the birth (calving) event
	- 1: Unassisted
	- 2: Easy pull / slight problem. One person (can be using chains and handles) but without the use of ropes / pulley.
	- 3: Difficult pull / two people pulling or one person plus moderate use of ropes / pulley
	- 4: Very difficult pull / large amount of force needed. Ropes / pulley essential for delivery of calf
	- 5: C-section or fetotomy
Dyst: (numeric, categorical) Occurrence of dystocia.
	- 0: No dystocia (Unassisted calving / parturition that is score 1 from calving difficult)
	- 1: Dystocia (Assisted calving / birth - Score 2 to 5 from calving difficult score)
CalfInfo: (string, categorical) Gender of the calf
	- M: Male
	- F: Female
	- T: Twins
	- S: Stillbirth
RFM: (numeric, categorical) Retained fetal membranes. Whether if the cow had failure to expel fetal membranes within 12 hours after parturition.
	- 1: Yes
	- 0: No
BCS5: (numeric) Body condition score. Visual evaluation of body fat reserves using a 5-point scale with 0.25-point increments where 1=very thin 5=very fat.
BCS5C: (string, categorical) BCS at 5 DIM. (Low, moderate, or high)
VLS: (numeric, categorical) Vaginal laceration due to birth (calving)
	- 0: No
	- 1: Yes and < 2 cm
	- 2: Yes > 2 cm
VLSC: (numeric, categorical) Vaginal laceration due to birth (calving)
	- 0: No VLS or < 2 cm
	- 1: Yes VLS > 2 cm)
Fevd0: (numeric, categorical) If the animal was or not febrile at the metritis diagnose day
	- 1: Yes
	- 0: No
Tempd0: (numeric) Rectal temperature (°C) measured on the day of metritis diagnosis
Cured: (numeric, categorical) If the animal cured up to 12 days after enrolment
	- 1: Cured
	- 0: Not cured
Daystocure: (numeric) Days from enrolment (metritis diagnosis) to cure
DIMcure: (numeric) Days in milk for cure (Days from the calving to the day that the cure was diagnosed)
Milkincrease5DIM: (numeric) Slope of the increase in milk production in the first 05 days of lactation
Milkincrease7DIM: (numeric) Slope of the increase in milk production in the first 07 days of lactation
Milkincrease9DIM: (numeric) Slope of the increase in milk production in the first 09 days of lactation
```

5. Missing attribute values

```
CalvDif: 5
Daystocure: 153
DIMcure: 153
Milkincrease5DIM: 87
Milkincrease7DIM: 75
Milkincrease9DIM: 68
```

